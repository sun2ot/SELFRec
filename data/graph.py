import numpy as np
import scipy.sparse as sp


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        """
        对给定的邻接矩阵进行规范化
        
        Args:
            adj_mat (scipy.sparse.coo_matrix): 图的邻接矩阵
        
        Returns:
            norm_adj_mat (scipy.sparse.coo_matrix): 规范化后的邻接矩阵
        """
        # 获取邻接矩阵的形状，用于后续判断矩阵是否为方阵
        shape = adj_mat.get_shape()
        # 计算邻接矩阵每行的和，用于构造度矩阵
        rowsum = np.array(adj_mat.sum(1))
        
        # 判断邻接矩阵是否为方阵
        if shape[0] == shape[1]:
            # 对于方阵，计算度矩阵的逆的平方根
            d_inv = np.power(rowsum, -0.5).flatten()
            # 将无穷大的值设为0，避免在度矩阵的逆中出现无穷大值
            d_inv[np.isinf(d_inv)] = 0.
            # 构造度矩阵的逆的平方根
            d_mat_inv = sp.diags(d_inv)
            # 进行图规范化操作
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            # 对于非方阵，计算度矩阵的逆
            d_inv = np.power(rowsum, -1).flatten()
            # 将无穷大的值设为0，避免在度矩阵的逆中出现无穷大值
            d_inv[np.isinf(d_inv)] = 0.
            # 构造度矩阵的逆
            d_mat_inv = sp.diags(d_inv)
            # 进行图规范化操作
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass
