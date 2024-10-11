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
import time
import os
from safetensors import safe_open

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
        self.do_save: bool = config['save']
        self.device = torch.device(f"cuda:{int(self.config['gpu_id'])}" if torch.cuda.is_available() else "cpu")
        Log.cli('Model', f'running on device {self.device}')
        self.model = XSimGCL_Encoder(self.data,
                                     self.emb_size, self.eps, self.n_layers, self.cl_layer,
                                     self.device, self.do_save, **kwargs)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        start_batch100_time = time.time()
        for epoch in range(self.maxEpoch):
            # 遍历每个批次的数据
            for n, batch_data in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                user_ids, pos_ids, neg_ids = batch_data

                # 获取推荐子图嵌入和对比学习子图嵌入
                rec_user_emb, rec_item_emb, \
                cl_user_emb, cl_item_emb, \
                fusion_user_embeddings, fusion_item_embeddings, user_pref_tensor = model(perturbed=True)

                # 根据批次数据获取用户的嵌入、正样本嵌入和负样本嵌入
                #* 这里看似字典形式获取，实则为索引，可参考下文predict()
                #? neg_ids 为多值时，形状如何？ -> (batch_size, n_negs, dim)
                user_emb, pos_item_emb, neg_item_embs = rec_user_emb[user_ids], rec_item_emb[pos_ids], rec_item_emb[neg_ids]
                fusion_user_emb, fusion_item_emb = None, None
                if fusion_user_embeddings is not None and fusion_item_embeddings is not None:
                    fusion_user_emb, fusion_item_emb = fusion_user_embeddings[user_ids], fusion_item_embeddings[pos_ids]

                #* 根据 neg_ids 取出对应中心性系数
                item_id_centrality = self.data.item_id_centrality
                neg_item_centralities = []
                for neg_id in neg_ids:
                    neg_item_centralities.append([item_id_centrality[id] for id in neg_id])
                # 负样本权重 (batch_size, 2*n_negs)
                neg_weights = torch.tensor(neg_item_centralities, dtype=torch.float, device=self.device)
                norm_neg_weights = F.normalize(neg_weights, p=2, dim=1)
                weight_neg_item_embs: torch.Tensor = norm_neg_weights.unsqueeze(-1) * neg_item_embs  # [batch_size, 2*n_negs, dim]

                # #todo 文本模态引导负样本采样
                if self.data.text_modal:
                    # 获取用户偏好
                    user_pref: torch.Tensor = user_pref_tensor[user_ids]  # (batch_siza, dim)
                    # 计算相似度
                    similarity = torch.bmm(weight_neg_item_embs, user_pref.unsqueeze(-1)).squeeze(-1)  # (batch_size, 2*n_negs)
                    # 排序索引
                    sorted_indices = torch.argsort(similarity, descending=True, dim=-1)
                    # 最小值索引
                    lowest_sim_indices = sorted_indices[:, -self.n_negs]  # (batch_size, n_negs)
                    # 相似度最低样本组成硬负样本  # (batch_size, n_negs, emb_size)
                    weight_neg_item_embs = neg_item_embs[torch.arange(len(neg_ids), device=self.device), lowest_sim_indices]

                # 计算推荐损失
                rec_loss1 = bpr_loss_w(user_emb, pos_item_emb, weight_neg_item_embs)
                # rec_loss2 = bpr_loss_w(user_emb, pos_item_emb, neg_item_embs, norm_neg_weights)

                # 计算对比学习损失
                # cl_loss = self.cl_rate * self.cal_cl_loss([user_ids, pos_ids], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_ids, pos_ids], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)

                # 计算批次总损失
                cross_modal_loss = None
                if self.data.image_modal or self.data.text_modal:
                    # 跨模态对比学习损失(注意id的选择与对比视图应对应)
                    # cross_modal_loss = self.cl_rate * self.cl_loss([user_ids, pos_ids, neg_ids], rec_user_emb, fusion_user_embeddings, rec_item_emb, fusion_item_embeddings)
                    batch_loss = rec_loss1 + l2_reg_loss(self.reg, user_emb, pos_item_emb, fusion_user_emb, fusion_item_emb) + cl_loss
                else:
                    batch_loss = rec_loss1 + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss

                # 梯度清零
                optimizer.zero_grad()
                # 反向传播
                batch_loss.backward()
                # 优化器更新参数
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    end_batch100_time = time.time()
                    elapsed_time = end_batch100_time - start_batch100_time
                    start_batch100_time = time.time()
                    if cross_modal_loss:
                        print(f"epoch: {epoch+1}, batch: {n}, time: {elapsed_time:.4f}s, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}, cross_modal_loss: {cross_modal_loss.item()}")
                    else:
                        print(f"epoch: {epoch+1}, batch: {n}, time: {elapsed_time:.4f}s, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}")
                    # print(f"epoch: {epoch+1}, batch: {n}, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}")

            # epoch结束，验证并更新最佳模型
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        # 最终更新用户嵌入和物品嵌入为最佳嵌入
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
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
            out = self.model()
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
    def __init__(self, data: Interaction, emb_size, eps, n_layers, cl_layer, device: torch.device, do_save: bool, **kwargs):
        Log.cli('XSimGCL_Encoder', 'init')
        super(XSimGCL_Encoder, self).__init__()
        self.device = device
        self.do_save = do_save
        self.model_name = kwargs.get('model_name')
        self.timestamp = kwargs.get('timestamp')

        self.data = data
        self.eps = eps  # epsilon -> CL Loss 超参数
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.cl_layer = cl_layer
        self.norm_adj = data.norm_adj
        self.param_dict = self._init_model()
        self.image_modal_flag, self.text_modal_flag = False, False
        self._init_multi_modal(self.do_save)
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj, device=self.device)

    def _init_model(self):
        """
        使用Xavier初始化模型的嵌入参数

        Returns:
            param_dict (nn.ParameterDict): 包含用户和物品嵌入的参数字典，模型的可学习参数
        """
        #? 这里学习的东西，是用户和项目最终的那个嵌入？还是将他们的编码映射成嵌入张量的参数？
        # 实例化Xavier均匀初始化方法，用于后续的嵌入初始化
        initializer = nn.init.xavier_uniform_
        param_dict = nn.ParameterDict({
            # 创建用户&项目的嵌入矩阵(size = 数量 x 嵌入尺寸)
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size, device=self.device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size, device=self.device))),
        })
        return param_dict


    def _init_multi_modal(self, do_save: bool=False):
        if self.data.image_modal:
            Log.cli('Model', f'📷 Loading image safetensors to {self.device} and project to {self.emb_size} dimensions')
            # 定义图像投影层
            image_projection = nn.Linear(512, self.emb_size, device=self.device)
            if do_save:
                os.makedirs(f"pth/{self.model_name}_{self.timestamp}", exist_ok=True)
                torch.save(image_projection.state_dict(), f'pth/{self.model_name}_{self.timestamp}/image.pth')
            origin_image_tensor = torch.empty(size=(self.data.item_num, 512), device=self.device)
            
            # 初始化预训练图像嵌入张量(按照训练集image_id排列)
            item2image: dict[str, list[str]] = {}
            with safe_open(self.data.image_modal['image_set'], 'pt', device=f"cuda:{self.device.index}") as image_safetensors: # type: ignore
                with open(self.data.image_modal['item2image'], 'r') as map_file:
                    for line in map_file:
                        item = line.strip().split(' ')[0]
                        images = line.strip().split(' ')[1:]
                        item2image[item] = images
                for idx, item in tqdm(enumerate(self.data.item), desc='item images'):
                    try:
                        origin_image_tensor[idx] = torch.mean(
                            torch.stack([image_safetensors.get_tensor(image) for image in item2image[item]]), dim=0
                        )
                    except Exception as e:
                        Log.catch(e, item, 'item2photo emb project')
                        exit(-1)
                self.param_dict['image_embs_tensor'] = image_projection(origin_image_tensor)
                self.image_modal_flag = True

        if self.data.text_modal:
            Log.cli('Model', f'📒 Loading text safetensors to {self.device} and project to {self.emb_size} dimensions')
            # 定义文本投影层
            item_text_projection = nn.Linear(1024, self.emb_size, device=self.device)
            user_pref_projection = nn.Linear(1024, self.emb_size, device=self.device)

            if do_save:
                os.makedirs(f"pth/{self.model_name}_{self.timestamp}", exist_ok=True)
                torch.save(item_text_projection.state_dict(), f'pth/{self.model_name}_{self.timestamp}/item_text.pth')
                torch.save(user_pref_projection.state_dict(), f'pth/{self.model_name}_{self.timestamp}/user_pref.pth')

            # 初始化预训练文本嵌入张量(按照训练集item_id排列)
            origin_text_tensor = torch.empty(size=(self.data.item_num, 1024), device=self.device)
            origin_pref_tensor = torch.empty(size=(self.data.user_num, 1024), device=self.device)

            with safe_open(self.data.text_modal['item_text'], 'pt', device=f"cuda:{self.device.index}") as f1: # type: ignore
                for idx, item in tqdm(enumerate(self.data.item), desc='item text'):
                    origin_text_tensor[idx] = f1.get_tensor(item)
            with safe_open(self.data.text_modal['user_pref'], 'pt', device=f"cuda:{self.device.index}") as f2: # type: ignore
                for idx, user in tqdm(enumerate(self.data.user), desc='user pref'):
                    origin_pref_tensor[idx] = f2.get_tensor(user)
            
            self.param_dict['item_text_tensor'] = item_text_projection(origin_text_tensor)
            #! 这玩意不需要模型优化
            self.user_pref_tensor: torch.Tensor = user_pref_projection(origin_pref_tensor)
            self.text_modal_flag = True


    def forward(self, perturbed=False):
        """
        前向传播函数，用于计算用户和物品的嵌入向量

        Args:
            perturbed (bool): 是否对嵌入向量进行扰动

        Returns:
            如果perturbed为True，则返回user&item emb和扰动后的user&item per_emb
            
            否则返回user&item emb
        """
        #* 为解耦多模态实现, 暂考虑复用逻辑
        final_image_embeddings = None
        final_text_embeddings = None
        user_pref_tensor = None
        if hasattr(self, 'user_pref_tensor'):
            user_pref_tensor = self.user_pref_tensor.detach()

        if self.image_modal_flag:
            image_side_embs = torch.cat([self.param_dict['user_emb'], self.param_dict['image_embs_tensor']], 0)
            all_image_embeddings = []
            for k in range(self.n_layers):
                image_side_embs = torch.sparse.mm(self.sparse_norm_adj, image_side_embs)
                # if perturbed:
                #     random_noise = torch.rand_like(image_side_embs)
                #     image_side_embs += torch.sign(image_side_embs) * F.normalize(random_noise, dim=-1) * self.eps
                all_image_embeddings.append(image_side_embs)

            final_image_embeddings = torch.mean(torch.stack(all_image_embeddings, dim=1), dim=1)
            final_image_embeddings = F.leaky_relu(final_image_embeddings)
            final_image_embeddings = nn.Dropout(p=0.2)(final_image_embeddings)
            final_image_embeddings = F.normalize(final_image_embeddings, p=2)
        
        if self.text_modal_flag:
            text_side_embs = torch.cat([self.param_dict['user_emb'], self.param_dict['item_text_tensor']], 0)
            all_text_embeddings = []
            for k in range(self.n_layers):
                text_side_embs = torch.sparse.mm(self.sparse_norm_adj, text_side_embs)
                # if perturbed:
                #     random_noise = torch.rand_like(text_side_embs)
                #     text_side_embs += torch.sign(text_side_embs) * F.normalize(random_noise, dim=-1) * self.eps
                all_text_embeddings.append(text_side_embs)
            final_text_embeddings = torch.mean(torch.stack(all_text_embeddings, dim=1), dim=1)
            final_text_embeddings = F.leaky_relu(final_text_embeddings)
            final_text_embeddings = nn.Dropout(p=0.2)(final_text_embeddings)
            final_text_embeddings = F.normalize(final_text_embeddings, p=2)
        
        # 将用户和物品嵌入拼接在一起，形成初始的嵌入矩阵
        # (user_num, dim) || (item_num, dim) = (node_num, dim)
        ego_embeddings = torch.cat([self.param_dict['user_emb'], self.param_dict['item_emb']], 0)
        # 初始化一个列表，用于存储每一层的嵌入向量
        all_embeddings = []
        # 用于存储对比对比学习模块最终层的嵌入向量
        #! 尚未进行CL，只是经过两层LightGCN
        all_embeddings_cl = ego_embeddings
        # 对于每一层进行消息传递和聚合
        for k in range(self.n_layers):
            # 信息传播: 邻接矩阵 x 嵌入向量 -> torch.Size([node_num, node_num]) x torch.Size([node_num, dim])
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                # 为嵌入向量添加扰动
                # Returns a tensor with the same size as input 
                # that is filled with random numbers from a uniform distribution on the interval [0, 1)
                random_noise = torch.rand_like(ego_embeddings)
                # torch.sign returns a new tensor with the signs(1|-1|0) of the elements of input.
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps

            all_embeddings.append(ego_embeddings)

            # 选定对比学习所在层
            if k == self.cl_layer-1:
                all_embeddings_cl = ego_embeddings

        # 将所有层的嵌入向量堆叠成一个三维矩阵，然后在第二维度上取平均，得到最终的嵌入向量
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        # 将最终的嵌入向量分割成用户嵌入和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        
        # 模态融合v4/5: 拼接均值(取消原始嵌入融合, 单独构建多模态)
        fusion_user_embeddings, fusion_item_embeddings = None, None
        if final_image_embeddings is not None and final_text_embeddings is not None:  #* 两种模态
            image_side_user, all_image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            text_side_user, all_text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            # fusion_user_embeddings = torch.mean(torch.stack([user_all_embeddings, image_side_user, text_side_user], dim=0), dim=0)
            # fusion_item_embeddings = torch.mean(torch.stack([item_all_embeddings, all_image_embs, all_text_embs], dim=0), dim=0)
            fusion_user_embeddings = torch.mean(torch.stack([image_side_user, text_side_user], dim=0), dim=0)
            fusion_item_embeddings = torch.mean(torch.stack([all_image_embs, all_text_embs], dim=0), dim=0)
        elif final_image_embeddings is not None:  #* 图片模态
            image_side_user, all_image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            # fusion_user_embeddings = torch.mean(torch.stack([user_all_embeddings, image_side_user], dim=0), dim=0)
            # fusion_item_embeddings = torch.mean(torch.stack([item_all_embeddings, all_image_embs], dim=0), dim=0)
            fusion_user_embeddings = image_side_user
            fusion_item_embeddings = all_image_embs
        elif final_text_embeddings is not None:  #* 文本模态
            text_side_user, all_text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            # fusion_user_embeddings = torch.mean(torch.stack([user_all_embeddings, text_side_user], dim=0), dim=0)
            # fusion_item_embeddings = torch.mean(torch.stack([item_all_embeddings, all_text_embs], dim=0), dim=0)
            fusion_user_embeddings = text_side_user
            fusion_item_embeddings = all_text_embs

        # else:
        #     Log.cli('Forward', 'No fusion embeddings.')
        
        if perturbed:
            # return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl, fusion_user_embeddings, fusion_item_embeddings, user_pref_tensor
        return user_all_embeddings, item_all_embeddings
