import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import pickle

class Interaction(Data,Graph):
    """ui交互图封装"""
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self,conf,training,test)

        self.user = {}  # 用户id -> 编码
        self.item = {}  # 物品id -> 编码
        # id映射
        self.id2user = {}
        self.id2item = {}
        # 数据集(双向嵌套字典)
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        # 用户和项目的数量
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        # 交互二分图邻接矩阵
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        # 交互邻接矩阵
        self.interaction_mat = self.__create_sparse_interaction_matrix()
        # popularity_user = {}
        # for u in self.user:
        #     popularity_user[self.user[u]] = len(self.training_set_u[u])
        # popularity_item = {}
        # for u in self.item:
        #     popularity_item[self.item[u]] = len(self.training_set_i[u])


    def __generate_set(self):
        """
        生成用户、物品和评分的集合
        """
        # 遍历训练数据，为每个用户和物品分配ID，并构建用户-物品评分矩阵
        for entry in self.training_data:
            user, item, rating = entry
            # 如果用户不在字典中，为其分配一个新的ID
            if user not in self.user:
                self.user[user] = len(self.user)  # 直接以长度编码(0,1,2,3,...)
                self.id2user[self.user[user]] = user
            # 物品编码同上
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            # 生成评分记录(嵌套dict)
            #! 根据库中的yelp数据集，评分全部为1，这是否会有影响？
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
            
        # 遍历测试数据，只为训练数据中已有的用户和物品添加评分记录至测试集
        #* 只处理训练集中存在的数据，这一点很重要
        for entry in self.test_data:
            user, item, rating = entry
            # 忽略测试数据中未在训练数据出现的用户或物品
            if user not in self.user or item not in self.item:
                continue
            # 将评分记录添加到测试集的用户-物品评分矩阵中，并记录测试集中的所有物品
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        """
        创建并返回一个稀疏的二分图邻接矩阵
        
        Args:
            self_connection (bool): 如果为True，则邻接矩阵的对角线添加自连接
        
        Returns:
            scipy.sparse.csr_matrix: 稀疏的邻接矩阵，形状为(user number + item number, user number + item number)
        """
        # 计算图中节点的总数，包括用户和项目
        n_nodes = self.user_num + self.item_num  # 69716
        # 获取训练数据中用户/项目的id(也即索引)分别作为行/列索引
        # self.training_data -> List[[user, item, float(weight)], [...]]
        row_idx = [self.user[pair[0]] for pair in self.training_data]  # len = 1237259 -> (self.training_data.size)
        col_idx = [self.item[pair[1]] for pair in self.training_data]  # len = 1237259
        # 将用户和项目的索引转换为NumPy数组
        user_np = np.array(row_idx)  # (1237259,)
        item_np = np.array(col_idx)  # (1237259,)
        # 创建一个与用户索引数组相同形状的数组，填充值为1，用于后续创建加权矩阵
        ratings = np.ones_like(user_np, dtype=np.float32)  # (1237259,)
        # 创建一个稀疏的CSR格式邻接矩阵，考虑到用户和项目之间的边
        #* (ratings, (user_np, item_np + self.user_num)) 分别作为非零元素值和对应位置
        #* item_np + self.user_num 表示将物品 ID 偏移了 self.user_num，使得物品 ID 与用户 ID 不会重叠
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.user_num)), shape=(n_nodes, n_nodes), dtype=np.float32)
        # 通过添加其转置矩阵来创建对称的邻接矩阵
        adj_mat = tmp_adj + tmp_adj.T
        # 如果self_connection参数为True，则在邻接矩阵的对角线上添加自连接
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0]+adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),shape=(n_nodes, n_nodes),dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        创建一个稀疏的user-item交互矩阵

        Returns:
            一个稀疏的邻接矩阵，形状为(user number, item number)
        """
        row, col, entries = [], [], []
        # self.training_data -> List[[user, item, float(weight)], [...]]
        for pair in self.training_data:
            row.append(self.user[pair[0]])
            col.append(self.item[pair[1]])
            entries.append(1.0)
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        'whether user u rated item i'
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        'whether user is in training set'
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        """
        获取用户u的评分信息

        Args:
            u: 用户ID，表示我们想要查询评分信息的用户

        Returns:
            (list1, list2) (tuple): (用户u评价过的所有物品的ID, 对应物品的评分)
        """
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        # print vec
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        # print vec
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            # print vec
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m
