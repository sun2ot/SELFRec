import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import l2_reg_loss, bpr_loss_w, cl_loss
from util.logger import Log
from data.ui_graph import Interaction
from tqdm import tqdm
import time
import os
from safetensors import safe_open
from typing import Optional
from dataclasses import dataclass

#todo 测试 torch.jit 的加速效果
bpr_script = torch.jit.script(bpr_loss_w)
l2_reg_script = torch.jit.script(l2_reg_loss)
cl_script = torch.jit.script(cl_loss)


@dataclass
class Emb():
    user_embs: torch.Tensor = torch.ones(1)  # 无意义, 仅作占位符tensor
    item_embs: torch.Tensor = torch.ones(1)
    user_embs_cl: torch.Tensor = torch.ones(1)
    item_embs_cl: torch.Tensor = torch.ones(1)
    user_pref_embs: Optional[torch.Tensor] = None
    image_embs: Optional[torch.Tensor] = None
    image_embs_cl: Optional[torch.Tensor] = None
    text_embs: Optional[torch.Tensor] = None
    text_embs_cl: Optional[torch.Tensor] = None


class XSimGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(XSimGCL, self).__init__(conf, training_set, test_set, **kwargs)

        self.model_config = self.config['XSimGCL']
        self.n_negs = int(self.model_config['n_negs'])
        self.temp = float(self.model_config['tau'])
        self.cl_rate = float(self.model_config['lambda'])
        self.device = torch.device(f"cuda:{int(self.config['gpu_id'])}" if torch.cuda.is_available() else "cpu")
        self.kwargs = kwargs
    
    def build(self):
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.model_config, self.device, self.kwargs)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        train_start_time = time.time()
        start_batch100_time = time.time()
        for epoch in range(self.maxEpoch):
            # 遍历每个批次的数据
            for n, batch_data in enumerate(next_batch_pairwise(self.data, self.batch_size, self.n_negs)):
                user_ids, pos_ids, neg_ids = batch_data

                # 获取推荐子图嵌入和对比学习子图嵌入
                embs = model.forward(perturbed=True)
                rec_user_emb, rec_item_emb = embs.user_embs, embs.item_embs
                cl_user_emb, cl_item_emb = embs.user_embs_cl, embs.item_embs_cl
                image_embs, image_embs_cl = embs.image_embs, embs.image_embs_cl
                text_embs, text_embs_cl = embs.text_embs, embs.text_embs_cl
                user_pref_tensor = embs.user_pref_embs

                # 根据批次数据获取用户的嵌入、正样本嵌入和负样本嵌入
                #* 这里看似字典形式获取，实则为索引，可参考下文predict()
                user_emb, pos_item_emb, neg_item_embs = rec_user_emb[user_ids], rec_item_emb[pos_ids], rec_item_emb[neg_ids]

                #* 根据 neg_ids 取出对应中心性系数
                item_id_centrality = self.data.item_id_centrality
                neg_item_centralities = []
                for neg_id in neg_ids:
                    neg_item_centralities.append([item_id_centrality[id] for id in neg_id])
                # 负样本权重 (batch_size, 2*n_negs)
                neg_weights = torch.tensor(neg_item_centralities, dtype=torch.float, device=self.device)
                norm_neg_weights = F.normalize(neg_weights, p=2, dim=1)
                weight_neg_item_embs: torch.Tensor = norm_neg_weights.unsqueeze(-1) * neg_item_embs  # [batch_size, 2*n_negs, dim]

                # 用户偏好引导负样本采样
                if user_pref_tensor is not None:
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
                rec_loss1 = bpr_script(user_emb, pos_item_emb, weight_neg_item_embs) # type: ignore

                # 计算对比学习损失
                user_cl_loss = self.cl_rate * cl_script(user_ids, rec_user_emb, cl_user_emb, self.temp, self.device) # type: ignore
                item_cl_loss = self.cl_rate * cl_script(pos_ids, rec_item_emb, cl_item_emb, self.temp, self.device) # type: ignore
                ui_cl_loss = user_cl_loss + item_cl_loss

                # image_cl_loss = self.cl_rate * cl_script(pos_ids, image_embs, image_embs_cl, self.temp, self.device) # type: ignore
                # text_cl_loss = self.cl_rate * cl_script(pos_ids, text_embs, text_embs_cl, self.temp, self.device) # type: ignore

                # total_cl_loss = ui_cl_loss + image_cl_loss + text_cl_loss

                # 计算批次总损失
                if self.data.image_modal or self.data.text_modal:
                    batch_loss = rec_loss1 + l2_reg_script(self.reg, [user_emb, pos_item_emb], self.device) + ui_cl_loss # type: ignore
                else:
                    batch_loss = rec_loss1 + l2_reg_script(self.reg, [user_emb, pos_item_emb], self.device) + ui_cl_loss # type: ignore

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
                    print(f"epoch: {epoch+1}, batch: {n}, time: {elapsed_time:.4f}s, rec_loss: {rec_loss1.item()}, cl_loss: {ui_cl_loss.item()}") # type: ignore
                    # print(f"epoch: {epoch+1}, batch: {n}, rec_loss: {rec_loss1.item()}, cl_loss: {cl_loss.item()}")

            # epoch结束，验证并更新最佳模型
            with torch.no_grad():
                embs = self.model.forward()
                self.user_emb, self.item_emb = embs.user_embs, embs.item_embs
            self.fast_evaluation(epoch)
            
            if self.early_stop == 10:
                break  # 连续5次性能未提升, 结束训练s
        
        # 最终更新用户嵌入和物品嵌入为最佳嵌入
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        """保存最佳用户嵌入和物品嵌入"""
        # with torch.no_grad():
        #     out = self.model()
        #     self.best_user_emb, self.best_item_emb = out[0], out[1]
        self.best_user_emb, self.best_item_emb = self.user_emb, self.item_emb

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
    def __init__(self, data: Interaction, emb_size: int, model_config: dict, device: torch.device, kwargs):
        super(XSimGCL_Encoder, self).__init__()

        self.data = data
        self.emb_size = emb_size
        self.device = device
        self.model_name = kwargs.get('model_name')
        self.timestamp = kwargs.get('timestamp')

        self.eps = model_config['eps']
        self.n_layer = model_config['n_layer']
        self.cl_layer = model_config['cl_layer']

        self.norm_adj = self.data.norm_adj
        self.param_dict = self._init_model()
        self.image_modal_flag, self.text_modal_flag = False, False
        self._init_multi_modal()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj, device=self.device)

    def _init_model(self):
        """
        使用Xavier初始化模型的嵌入参数

        Returns:
            param_dict (nn.ParameterDict): 包含用户和物品嵌入的参数字典，模型的可学习参数
        """
        # 实例化Xavier均匀初始化方法，用于后续的嵌入初始化
        initializer = nn.init.xavier_uniform_
        param_dict = nn.ParameterDict({
            # 创建用户&项目的嵌入矩阵(size = 数量 x 嵌入尺寸)
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size, device=self.device))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size, device=self.device))),
        })
        return param_dict


    def _init_multi_modal(self):
        image_modal = self.data.image_modal
        text_modal = self.data.text_modal
        user_pref = self.data.user_pref

        if image_modal:
            Log.cli('Model', f'📷 Loading image safetensors to {self.device} and project to {self.emb_size} dimensions')
            
            # 图像投影层
            image_projection = nn.Linear(512, self.emb_size, device=self.device)
            if image_modal['pre_trained']['enable']:
                try:
                    # 加载预训练参数
                    image_pth = image_modal['pre_trained']['image_pth']
                    image_projection.load_state_dict(torch.load(image_pth))
                except Exception as e:
                    Log.catch(e, 'image_modal', '_init_multi_modal')
                    exit(-1)
            else:
                if image_modal['pre_trained']['save']:
                    path = image_modal['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(image_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/image.pth')
            
            # 初始化预训练图像嵌入张量(按照训练集image_id排列)
            origin_image_tensor = torch.empty(size=(self.data.item_num, 512), device=self.device)
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

        if text_modal:
            Log.cli('Model', f'📒 Loading text safetensors to {self.device} and project to {self.emb_size} dimensions')
            # 文本投影层
            item_text_projection = nn.Linear(1024, self.emb_size, device=self.device)
            if text_modal['pre_trained']['enable']:
                try:
                    # 加载预训练参数
                    item_text_pth = text_modal['pre_trained']['item_text_pth']
                    item_text_projection.load_state_dict(torch.load(item_text_pth))
                except Exception as e:
                    Log.catch(e, 'text_modal', '_init_multi_modal')
                    exit(-1)
            else:
                if text_modal['pre_trained']['save']:
                    path = image_modal['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(item_text_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/item_text.pth')

            # 初始化预训练文本嵌入张量(按照训练集item_id排列)
            origin_text_tensor = torch.empty(size=(self.data.item_num, 1024), device=self.device)

            with safe_open(self.data.text_modal['item_text'], 'pt', device=f"cuda:{self.device.index}") as f1: # type: ignore
                for idx, item in tqdm(enumerate(self.data.item), desc='item text'):
                    origin_text_tensor[idx] = f1.get_tensor(item)
            
            self.param_dict['item_text_tensor'] = item_text_projection(origin_text_tensor)
            self.text_modal_flag = True
        
        if user_pref:
            Log.cli('Model', f'📒 Loading pref safetensors to {self.device} and project to {self.emb_size} dimensions')
            user_pref_projection = nn.Linear(1024, self.emb_size, device=self.device)
            if user_pref['pre_trained']['enable']:
                try:
                    user_pref_pth = user_pref['pre_trained']['user_pref_pth']
                    user_pref_projection.load_state_dict(torch.load(user_pref_pth))
                except Exception as e:
                    Log.catch(e, 'text_modal', '_init_multi_modal')
                    exit(-1) 
            else:
                if user_pref['pre_trained']['save']:
                    path = user_pref['pre_trained']['save_path']
                    os.makedirs(f"{path}/{self.model_name}_{self.timestamp}", exist_ok=True)
                    torch.save(user_pref_projection.state_dict(), f'{path}/{self.model_name}_{self.timestamp}/user_pref.pth')
            
            origin_pref_tensor = torch.empty(size=(self.data.user_num, 1024), device=self.device)
            with safe_open(self.data.user_pref['pref_embs'], 'pt', device=f"cuda:{self.device.index}") as f2: # type: ignore
                for idx, user in tqdm(enumerate(self.data.user), desc='user pref'):
                    origin_pref_tensor[idx] = f2.get_tensor(user)
            
            #! 这玩意不需要模型优化
            self.user_pref_tensor: torch.Tensor = user_pref_projection(origin_pref_tensor)

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
        final_image_embeddings, final_text_embeddings = None, None
        
        # user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl, user_pref_tensor
        embs = Emb(
            user_embs = self.param_dict['user_emb'],
            item_embs = self.param_dict['item_emb']
        )
        assert embs.user_embs is not None
        assert embs.item_embs is not None
        
        if hasattr(self, 'user_pref_tensor'):
            embs.user_pref_embs = self.user_pref_tensor.detach()

        if self.image_modal_flag:
            embs.image_embs = self.param_dict['image_embs_tensor']
            assert embs.image_embs is not None
            image_side_embs = torch.cat([embs.user_embs, embs.image_embs], 0)

            all_image_embeddings = []
            for k in range(self.n_layer):  #* 图像模态传播
                image_side_embs = torch.sparse.mm(self.sparse_norm_adj, image_side_embs)
                # if perturbed:
                #     # 为嵌入向量添加扰动
                #     random_noise = torch.rand_like(image_side_embs)
                #     image_side_embs += torch.sign(image_side_embs) * F.normalize(random_noise, dim=-1) * self.eps
                all_image_embeddings.append(image_side_embs)
                # if k == self.cl_layer-1:
                #     _, embs.image_embs_cl = torch.split(image_side_embs, [self.data.user_num, self.data.item_num])

            final_image_embeddings = torch.mean(torch.stack(all_image_embeddings, dim=1), dim=1)
            final_image_embeddings = F.leaky_relu(final_image_embeddings)
            final_image_embeddings = nn.Dropout(p=0.2)(final_image_embeddings)
            final_image_embeddings = F.normalize(final_image_embeddings, p=2)
        
        if self.text_modal_flag:
            embs.text_embs = self.param_dict['item_text_tensor']
            assert embs.text_embs is not None
            text_side_embs = torch.cat([embs.user_embs, embs.text_embs], 0)

            all_text_embeddings = []
            for k in range(self.n_layer):  #* 文本模态传播
                text_side_embs = torch.sparse.mm(self.sparse_norm_adj, text_side_embs)
                # if perturbed:
                #     # 为嵌入向量添加扰动
                #     random_noise = torch.rand_like(text_side_embs)
                #     text_side_embs += torch.sign(text_side_embs) * F.normalize(random_noise, dim=-1) * self.eps
                all_text_embeddings.append(text_side_embs)
                # if k == self.cl_layer-1:
                #     _, embs.text_embs_cl = torch.split(text_side_embs, [self.data.user_num, self.data.item_num])
            
            final_text_embeddings = torch.mean(torch.stack(all_text_embeddings, dim=1), dim=1)
            final_text_embeddings = F.leaky_relu(final_text_embeddings)
            final_text_embeddings = nn.Dropout(p=0.2)(final_text_embeddings)
            final_text_embeddings = F.normalize(final_text_embeddings, p=2)
        
        #* 模态融合v6: 先处理多模态, 然后融合, 最后加噪对比
        if final_image_embeddings is not None and final_text_embeddings is not None:  #* 两种模态
            image_side_user, image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            text_side_user, text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            fusion_user_embeddings = torch.mean(torch.stack([embs.user_embs, image_side_user, text_side_user], dim=0), dim=0)
            fusion_item_embeddings = torch.mean(torch.stack([embs.item_embs, image_embs, text_embs], dim=0), dim=0)
            joint_embeddings = torch.cat([fusion_user_embeddings, fusion_item_embeddings], dim=0)
        elif final_image_embeddings is not None:  #* 图片模态
            image_side_user, image_embs = torch.split(final_image_embeddings, [self.data.user_num, self.data.item_num])
            fusion_user_embeddings = torch.mean(torch.stack([embs.user_embs, image_side_user], dim=0), dim=0)
            fusion_item_embeddings = torch.mean(torch.stack([embs.item_embs, image_embs], dim=0), dim=0)
            joint_embeddings = torch.cat([fusion_user_embeddings, fusion_item_embeddings], dim=0)
        elif final_text_embeddings is not None:  #* 文本模态
            text_side_user, text_embs = torch.split(final_text_embeddings, [self.data.user_num, self.data.item_num])
            fusion_user_embeddings = torch.mean(torch.stack([embs.user_embs, text_side_user], dim=0), dim=0)
            fusion_item_embeddings = torch.mean(torch.stack([embs.item_embs, text_embs], dim=0), dim=0)
            joint_embeddings = torch.cat([fusion_user_embeddings, fusion_item_embeddings], dim=0)
        else:
            joint_embeddings = torch.cat([embs.user_embs, embs.item_embs], 0)
        
        # 初始化一个列表，用于存储每一层的嵌入向量
        all_embeddings = []
        # 用于存储对比对比学习模块最终层的嵌入向量
        all_embeddings_cl = joint_embeddings
        # 对于每一层进行消息传递和聚合
        for k in range(self.n_layer):
            # 信息传播: 邻接矩阵 x 嵌入向量 -> torch.Size([node_num, node_num]) x torch.Size([node_num, dim])
            joint_embeddings = torch.sparse.mm(self.sparse_norm_adj, joint_embeddings)
            if perturbed:
                # 为嵌入向量添加扰动
                # Returns a tensor with the same size as input 
                # that is filled with random numbers from a uniform distribution on the interval [0, 1)
                random_noise = torch.rand_like(joint_embeddings)
                # torch.sign returns a new tensor with the signs(1|-1|0) of the elements of input.
                joint_embeddings += torch.sign(joint_embeddings) * F.normalize(random_noise, dim=-1) * self.eps

            all_embeddings.append(joint_embeddings)

            # 选定对比学习所在层
            if k == self.cl_layer-1:
                all_embeddings_cl = joint_embeddings

        # 将所有层的嵌入向量堆叠成一个三维矩阵，然后在第二维度上取平均，得到最终的嵌入向量
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)

        # 将最终的嵌入向量分割成用户嵌入和物品嵌入
        embs.user_embs, embs.item_embs = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        embs.user_embs_cl, embs.item_embs_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        
        if perturbed:
            return embs
        return embs
