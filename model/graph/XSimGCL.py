import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE, bpr_loss_w
from util.logger import Log
from data.ui_graph import Interaction
from tqdm import tqdm

#todo 测试 torch.jit 的加速效果
bpr_loss_script = torch.jit.script(bpr_loss)


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(XSimGCL, self).__init__(conf, training_set, test_set, **kwargs)
        config = self.config['XSimGCL']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.n_negs = int(config['n_negs'])
        self.cl_layer = int(config['cl_layer'])
        self.device = torch.device(f"cuda:{int(self.config['gpu_id'])}" if torch.cuda.is_available() else "cpu")
        Log.cli('Model', f'running on device {self.device}')
        self.model = XSimGCL_Encoder(self.data,
                                     self.emb_size, self.eps, self.n_layers, self.cl_layer,
                                     self.device)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            # 遍历每个批次的数据
            for n, batch_data in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                user_ids, pos_ids, neg_ids = batch_data

                # 获取推荐子图嵌入和对比学习子图嵌入
                # rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(perturbed=True)
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb, image_side_user, text_side_user = model(perturbed=True)
                # 根据批次数据获取用户的嵌入、正样本嵌入和负样本嵌入
                #* 这里看似字典形式获取，实则为索引，可参考下文predict()
                #? neg_ids 为多值时，形状如何？ -> (batch_size, n_negs, dim)
                user_emb, pos_item_emb, neg_item_embs = rec_user_emb[user_ids], rec_item_emb[pos_ids], rec_item_emb[neg_ids]

                #* 根据 neg_ids 取出对应中心性系数
                item_id_centrality = self.data.item_id_centrality
                neg_item_centralities = []
                for neg_id in neg_ids:
                    neg_item_centralities.append([item_id_centrality[id] for id in neg_id])
                # 负样本权重 (batch_size, n_negs)
                neg_weights = torch.tensor(neg_item_centralities, dtype=torch.float, device=self.device)
                norm_neg_weights = F.normalize(neg_weights, p=2, dim=1)

                #todo 文本模态引导负样本采样
                # 初始化
                hard_neg_item_embs = torch.empty(size=(len(neg_ids), self.emb_size), dtype=torch.float, device=self.device)
                # 获取用户偏好
                user_pref_emb = self.data.user_pref_tensor[user_ids].unsqueeze(1)  # (batch_size, 1, emb_size)
                # 计算相似度
                similarity = F.cosine_similarity(user_pref_emb, norm_neg_weights.unsqueeze(-1) * neg_item_embs, dim=2)  # (batch_size, n_negs)
                # 排序索引
                sorted_indices = torch.argsort(similarity, descending=True, dim=-1)
                # 最小值索引
                lowest_sim_indices = sorted_indices[:, -self.n_negs]  # (batch_size, n_negs)
                # 相似度最低样本组成硬负样本  # (batch_size, n_negs/2, emb_size)
                hard_neg_item_embs = neg_item_embs[torch.arange(len(neg_ids), device=self.device), lowest_sim_indices]

                # 计算推荐损失
                rec_loss1 = bpr_loss_w(user_emb, pos_item_emb, hard_neg_item_embs)
                # rec_loss2 = bpr_loss_w(user_emb, pos_item_emb, neg_item_embs, norm_neg_weights)

                # 计算对比学习损失
                cl_loss = self.cl_rate * self.cal_cl_loss([user_ids, pos_ids], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)

                # 跨模态对比学习损失(注意id的选择与对比视图应对应)
                cross_modal_loss1 = self.cl_rate * self.cross_modal_loss(user_ids, rec_user_emb, image_side_user)
                cross_modal_loss2 = self.cl_rate * self.cross_modal_loss(user_ids, rec_user_emb, text_side_user)
                cross_modal_loss = cross_modal_loss1 + cross_modal_loss2
                
                # 计算批次总损失
                # batch_loss = rec_loss2 + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                batch_loss = rec_loss1 + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss + cross_modal_loss
                # batch_loss = rec_loss2 + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss + cross_modal_loss

                # 梯度清零
                optimizer.zero_grad()
                # 反向传播
                batch_loss.backward()
                # 优化器更新参数
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print(f"epoch: {epoch+1}, batch: {n}, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}, cross_modal_loss: {cross_modal_loss.item()}")
                    # print(f"epoch: {epoch+1}, batch: {n}, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}")
            
            # epoch结束，验证并更新最佳模型
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        # 最终更新用户嵌入和物品嵌入为最佳嵌入
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx: list[list[int]], user_view1, user_view2, item_view1, item_view2):
        """
        对比学习损失函数
        
        Args:
            idx (list): [user_idx_list, pos_idx_list]
            user_view1: rec_user_emb
            user_view2: cl_user_emb
            item_view1:rec_item_emb
            item_view2: cl_item_emb
        
        Returns:
            user_cl_loss + item_cl_loss
        """
        # 确定唯一user/item索引 (u_idx ==/!= idx[0])
        u_idx = torch.unique(torch.tensor(idx[0], dtype=torch.long, device=self.device))
        i_idx = torch.unique(torch.tensor(idx[1], dtype=torch.long, device=self.device))
        # 使用InfoNCE损失函数计算user/item的对比损失
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def cross_modal_loss(self, idx: list[int], modal_view1: torch.Tensor, modal_view2: torch.Tensor):
        """(测试)跨模态对比学习损失

        Args:
            idx (list[int]): 正样本索引
            modal_view1 (torch.Tensor): 模态1
            modal_view2 (torch.Tensor): 模态2
        """
        idx_tensor = torch.unique(torch.tensor(idx, dtype=torch.long, device=self.device))
        cl_loss = InfoNCE(modal_view1[idx_tensor], modal_view2[idx_tensor], self.temp)
        return cl_loss


    def save(self):
        with torch.no_grad():
            out = self.model.forward()
            self.best_user_emb, self.best_item_emb = out[0], out[1]

    def predict(self, u):
        """
        根据用户ID预测其对所有物品的评分

        Args:
            u: user

        Returns:
            numpy数组，包含该用户对所有物品的预测评分 [score1, score2, ...]
        """
        # 获取用户的ID，确保u是用户在数据集中的唯一标识形式
        user_id = self.data.get_user_id(u)
        # 计算用户u对所有物品的预测评分: 用户嵌入和物品嵌入转置的乘积
        # self.user_emb -> torch.Size([31668, 64]), self.item_emb -> torch.Size([38048, 64])
        #* user_emb就是一个tensor，但是由于dataset处理的原因，user_id就是顺次排下来的数字(索引)
        #* 因此可以将user_id作为索引使用，以形似dict的方式获取对应user的emd
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))  # torch.Size([38048])
        # 将评分从GPU内存转回CPU内存，并转换为NumPy数组形式，以便后续处理或输出
        # 这一步转换是因为PyTorch张量通常在GPU上进行计算，而NumPy操作在CPU上
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    """
    XSimGCL 模型本体
    """
    def __init__(self, data: Interaction, emb_size, eps, n_layers, cl_layer, device: torch.device):
        super(XSimGCL_Encoder, self).__init__()
        self.device = device
        self.data = data
        self.eps = eps  # epsilon -> CL Loss 超参数
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.cl_layer = cl_layer
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj, device=self.device)

    def _init_model(self):
        """
        使用Xavier初始化模型的嵌入参数

        Returns:
            embedding_dict (nn.ParameterDict): 包含用户和物品嵌入的参数字典，这些嵌入是模型的可学习参数
        """
        #? 这里学习的东西，是用户和项目最终的那个嵌入？还是将他们的编码映射成嵌入张量的参数？
        # 实例化Xavier均匀初始化方法，用于后续的嵌入初始化
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            # 创建用户&项目的嵌入矩阵(size = 数量 x 嵌入尺寸)
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size, device=self.device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size, device=self.device))),
            'image_emb': nn.Parameter(self.data.image_embs_tensor),  # (item_num, dim)
            'text_emb': nn.Parameter(self.data.item_text_tensor),  # (item_num, dim)
            'fusion_weight': nn.Parameter(F.softmax(torch.randn(3, device=self.device), dim=0)) # (3)
        })

        # 图像模态融合v1
        # if self.data.image_embs is not None:
        #     #* 直接在GPU上初始化，否则会因频繁数据传输导致速度极慢
        #     item_embeddings = torch.zeros((self.data.item_num, self.emb_size), device=self.device)
        #     alpha = 0.5  # item 模态融合权重
        #     Log.cli('Model init', '📷 Image-modal fusion')
        #     for iid, image_tensor in enumerate(embedding_dict['item_emb']):
        #         try:
        #             item_embeddings[iid] = alpha * image_tensor + (1-alpha) * self.data.image_embs_tensor[iid]
        #         except Exception as e:
        #             Log.catch(exception=e, position=str(iid), subject='fusion')
        #             exit(-1)
        #     embedding_dict['item_emb'] = item_embeddings

        #! bug: core dumped 用户偏好增强
        # if self.data.user_pref_tensor is not None:
        #     user_embeddings = torch.zeros((self.data.user_num, self.emb_size), device=self.device)
        #     beta = 0.5  # 偏好融合权重
        #     Log.cli('Model init', '👨 User preference fusion')
        #     for uid, user_tensor in enumerate(embedding_dict['user_emb']):
        #         try:
        #             user_embeddings[uid] = beta * user_tensor + (1-beta) * self.data.user_pref_tensor[uid]
        #         except Exception as e:
        #             Log.catch(exception=e, position=str(uid), subject='fusion')
        #             exit(-1)
        #     embedding_dict['user_emb'] = user_embeddings
        
        return embedding_dict

    def forward(self, perturbed=False):
        """
        前向传播函数，用于计算用户和物品的嵌入向量

        Args:
            perturbed (bool): 是否对嵌入向量进行扰动

        Returns:
            如果perturbed为True，则返回user&item emb和扰动后的user&item per_emb
            
            否则返回user&item emb
        """
        # 将用户和物品嵌入拼接在一起，形成初始的嵌入矩阵
        # (user_num, dim) || (item_num, dim) = (node_num, dim)
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        image_side_embs = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['image_emb']], 0)
        text_side_embs = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['text_emb']], 0)
        # 初始化一个列表，用于存储每一层的嵌入向量
        all_embeddings = []
        all_image_embeddings = []
        all_text_embeddings = []
        # 用于存储对比对比学习模块最终层的嵌入向量
        #! 尚未进行CL，只是经过两层LightGCN
        all_embeddings_cl = ego_embeddings
        # 对于每一层进行消息传递和聚合
        for k in range(self.n_layers):
            # 信息传播: 邻接矩阵 x 嵌入向量 -> torch.Size([node_num, node_num]) x torch.Size([node_num, dim])
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            image_side_embs = torch.sparse.mm(self.sparse_norm_adj, image_side_embs)
            text_side_embs = torch.sparse.mm(self.sparse_norm_adj, text_side_embs)
            if perturbed:
                # 为嵌入向量添加扰动
                # Returns a tensor with the same size as input 
                # that is filled with random numbers from a uniform distribution on the interval [0, 1)
                random_noise = torch.rand_like(ego_embeddings)
                # torch.sign returns a new tensor with the signs(1|-1|0) of the elements of input.
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            
            image_side_embs = nn.LeakyReLU(negative_slope=0.01)(image_side_embs)
            text_side_embs = nn.LeakyReLU(negative_slope=0.01)(text_side_embs)

            image_side_embs = nn.Dropout(p=0.5)(image_side_embs)
            text_side_embs = nn.Dropout(p=0.5)(text_side_embs)

            norm_image_embs = F.normalize(image_side_embs, p=2, dim=1)
            norm_text_embs = F.normalize(text_side_embs, p=2, dim=1)
            
            all_embeddings.append(ego_embeddings)
            all_image_embeddings.append(norm_image_embs)
            all_text_embeddings.append(norm_text_embs)

            # 选定对比学习所在层
            if k == self.cl_layer-1:
                all_embeddings_cl = ego_embeddings

        # 将所有层的嵌入向量堆叠成一个三维矩阵，然后在第二维度上取平均，得到最终的嵌入向量
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        final_image_embeddings = torch.mean(torch.stack(all_image_embeddings, dim=1), dim=1)
        final_text_embeddings = torch.mean(torch.stack(all_text_embeddings, dim=1), dim=1)

        # 将最终的嵌入向量分割成用户嵌入和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        
        # 分割出传播后的 image/text embeddings
        image_side_user, all_image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
        text_side_user, all_text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])

        # 模态融合v3
        fusion_weight: torch.Tensor = self.embedding_dict['fusion_weight'].view(-1, 1, 1) # (3,1,1)
        item_all_embeddings = torch.stack([item_all_embeddings, all_image_embs, all_text_embs], dim=0)  # (3, item_num, dim)
        item_all_embeddings = (fusion_weight * item_all_embeddings).sum(dim=0)
        
        if perturbed:
            # return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl, image_side_user, text_side_user
        return user_all_embeddings, item_all_embeddings
