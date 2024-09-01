import torch
import numpy as np

class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        """
        将稀疏矩阵转换为PyTorch稀疏张量

        Args:
            X (scipy.sparse.coo_matrix): 稀疏矩阵
        
        Returns:
            torch.sparse_coo_tensor
        """
        # 将X转换为COO格式稀疏矩阵
        coo = X.tocoo()
        # 创建一个包含稀疏矩阵非零元素索引的张量
        # indices = torch.LongTensor([coo.row, coo.col])
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        # 创建一个包含稀疏矩阵非零元素值的浮点张量
        values = torch.from_numpy(coo.data).float()
        # 创建并返回一个PyTorch稀疏张量
        #! torch.sparse.FloatTensor已弃用
        # return torch.sparse.FloatTensor(indices, values, coo.shape)
        return torch.sparse_coo_tensor(indices, values, coo.shape)