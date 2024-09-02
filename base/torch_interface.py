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
        coords = np.array([coo.row, coo.col])
        i = torch.LongTensor(coords)
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
