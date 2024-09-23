import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}  # 用户 -> 用户id
        self.item = {}  # 物品 -> 物品id
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


    def __generate_set(self):
        """
        生成用户、物品和评分的集合
        """
        # 遍历训练数据，为每个用户和物品分配ID，并构建用户-物品评分矩阵
        for user, item, rating in self.training_data:
            # 如果用户不在字典中，为其分配一个新的ID
            if user not in self.user:
                user_id = len(self.user)
                self.user[user] = user_id  #* 直接以长度(索引)编码(0,1,2,3,...)
                self.id2user[user_id] = user
            # 物品编码同上
            if item not in self.item:
                item_id = len(self.item)
                self.item[item] = item_id
                self.id2item[item_id] = item
            # 生成评分记录(嵌套dict)
            #! 根据库中的yelp数据集，评分全部为1，这是否会有影响？
            # ans: 无影响，因为压根没有用上评分
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
            
        # 遍历测试数据，只为训练数据中已有的用户和物品添加评分记录至测试集
        #* 只处理训练集中存在的数据，这一点很重要
        for user, item, rating in self.test_data:
            # 忽略测试数据中未在训练数据出现的用户或物品
            # if user not in self.user or item not in self.item:
            #     continue
            # 将评分记录添加到测试集的用户-物品评分矩阵中，并记录测试集中的所有物品
            # self.test_set[user][item] = rating
            # self.test_set_item.add(item)

            # merge from upstream
            if user in self.user and item in self.item:
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
        # 将用户和项目的索引转换为NumPy数组
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])  # (1237259,)
        item_np = np.array([self.item[pair[1]] for pair in self.training_data])  # (1237259,)
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
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        """
        创建一个稀疏的user-item交互矩阵

        Returns:
            一个稀疏的邻接矩阵，形状为(user number, item number)
        """
        # self.training_data -> List[[user, item, float(weight)], [...]]
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.ones(len(row), dtype=np.float32)
        
        interaction_mat = sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)
        return interaction_mat



    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        """
        获取训练集大小
        
        Returns:
            (user number, item number, training data size)
        """
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

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
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
