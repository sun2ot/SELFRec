import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(XSimGCL, self).__init__(conf, training_set, test_set)
        config = self.config['XSimGCL']
        self.cl_rate = float(config['lambda'])
        self.eps = float(config['eps'])
        self.temp = float(config['tau'])
        self.n_layers = int(config['n_layer'])
        self.layer_cl = int(config['l_star'])
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers,self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            # 遍历每个批次的数据
            for n, batch_data in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                # 从批次数据中获取用户索引、正样本索引和负样本索引
                user_idx, pos_idx, neg_idx = batch_data
                # 获取推荐子图嵌入和对比学习子图嵌入
                #? 这个True参数哪来的, 推测为perturbed=True, 否则不会有四个返回值
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb  = model(True)
                # 根据批次数据获取用户的嵌入、正样本嵌入和负样本嵌入
                # 这里是根据位置索引获取的，不是 user/item id
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                # 计算推荐损失
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                # 计算对比学习损失
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx], rec_user_emb, cl_user_emb, rec_item_emb, cl_item_emb)
                # 计算批次总损失
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播
                batch_loss.backward()
                # 优化器更新参数
                optimizer.step()

                if n % 100 == 0 and n > 0:
                    print('training epoch:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
            
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
        # 确定唯一user/item索引
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        # 使用InfoNCE损失函数计算user/item的对比损失
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

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
        #* user_emb就是一个tensor，但是由于dataset处理的原因，user跟user_id都是顺次排下来的数字
        #* 因此可以将user_id作为索引使用，以形似dict的方式获取对应user的emd
        score = torch.matmul(self.user_emb[user_id], self.item_emb.transpose(0, 1))  # torch.Size([38048])
        # 将评分从GPU内存转回CPU内存，并转换为NumPy数组形式，以便后续处理或输出
        # 这一步转换是因为PyTorch张量通常在GPU上进行计算，而NumPy操作在CPU上
        return score.cpu().numpy()


class XSimGCL_Encoder(nn.Module):
    """
    XSimGCL 模型本体
    """
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps  # epsilon -> CL Loss 超参数
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

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
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
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
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        # 初始化一个列表，用于存储每一层的嵌入向量
        all_embeddings = []
        # 用于存储对比对比学习模块最终层的嵌入向量
        #! 尚未进行CL，只是经过两层LightGCN
        all_embeddings_cl = ego_embeddings
        # 对于每一层进行消息传递和聚合
        for k in range(self.n_layers):
            # 信息传播: 邻接矩阵 x 嵌入向量 -> torch.Size([69716, 69716]) x torch.Size([69716, 64])
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                # 为嵌入向量添加扰动
                # Returns a tensor with the same size as input 
                # that is filled with random numbers from a uniform distribution on the interval [0, 1)
                random_noise = torch.rand_like(ego_embeddings).cuda()
                # torch.sign returns a new tensor with the signs(1|-1|0) of the elements of input.
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            # 如果当前层是对比学习模块的最后一层，保存此时的嵌入向量
            if k == self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        # 将所有层的嵌入向量堆叠成一个三维矩阵，然后在第二维度上取平均，得到最终的嵌入向量
        final_embeddings = torch.stack(all_embeddings, dim=1)  # torch.Size([69716, 2, 64])
        final_embeddings = torch.mean(final_embeddings, dim=1)  # torch.Size([69716, 64])
        # 将最终的嵌入向量分割成用户嵌入和物品嵌入
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings,user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings
