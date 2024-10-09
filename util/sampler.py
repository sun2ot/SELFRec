import random
from random import shuffle, randint, choice, sample
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm

from data.ui_graph import Interaction
from util.logger import Log

random.seed(666)

# 日志文件名
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def next_batch_pairwise(data: Interaction, batch_size, n_negs):
    """
    生成用于训练的批量样本对

    Args:
        data (Interaction): 模型数据
        batch_size: 批量大小，决定每次迭代返回的样本数量
        n_negs: 每个用户的负样本数量

    Returns:
        yield: 每次yield出包含用户id、正样本id和负样本id列表
    """
    # neg_sample_log = Log(module='neg_sample', filename=f"neg_sample {current_time}")

    if n_negs <= 0:
        raise ValueError("n_negs must be greater than 0")
    
    # 获取并洗牌训练数据
    # SELFRec.py -> training_data -> [user, item, weight], [...], ...
    training_data = data.training_data
    # shuffle(training_data)  #* train data分割时已经打乱

    # 追踪当前处理到的数据位置
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        # 计算本批次的结束位置
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        # 收集本批次的用户和物品
        batch_users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        batch_items = [training_data[idx][1] for idx in range(ptr, batch_end)]

        ptr = batch_end
        # 初始化用户、正样本和负样本id列表
        u_ids: list[int] = []
        i_ids: list[int] = []
        j_ids: list[list[int]] = []

        # 从物品字典(物品->物品id)获取物品列表，用于负样本采样
        item_list = list(data.item.keys())

        # 为每个用户生成样本对
        for i, user in enumerate(batch_users):
            # 添加用户与正样本的一个索引对(user_id, pos_item_id)
            i_ids.append(data.get_item_id(batch_items[i]))
            u_ids.append(data.get_user_id(user))

            # 生成指定数量的负样本索引，并添加到j_ids
            neg_items: list[str] = []
            for _ in range(2*n_negs):
                # 从所有物品(训练集)随机获取一个负样本
                neg_item = choice(item_list)
                # 确保负样本不是用户的历史正样本
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                neg_items.append(neg_item)

            # 记录负采样结果
            # neg_sample_log.add(f"{user}: {','.join(neg_items)}")

            j_ids.append([data.get_item_id(item) for item in neg_items])

        # 返回本批次的用户、正样本和负样本id
        yield u_ids, i_ids, j_ids


def next_batch_pointwise(data,batch_size):
    training_data = data.training_data
    data_size = len(training_data)
    ptr = 0
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, y = [], [], []
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            y.append(1)
            for instance in range(4):
                item_j = randint(0, data.item_num - 1)
                while data.id2item[item_j] in data.training_set_u[user]:
                    item_j = randint(0, data.item_num - 1)
                u_idx.append(data.user[user])
                i_idx.append(item_j)
                y.append(0)
        yield u_idx, i_idx, y

# def next_batch_sequence(data, batch_size,n_negs=1):
#     training_data = data.training_set
#     shuffle(training_data)
#     ptr = 0
#     data_size = len(training_data)
#     item_list = list(range(1,data.item_num+1))
#     while ptr < data_size:
#         if ptr+batch_size<data_size:
#             end = ptr+batch_size
#         else:
#             end = data_size
#         seq_len = []
#         batch_max_len = max([len(s[0]) for s in training_data[ptr: end]])
#         seq = np.zeros((end-ptr, batch_max_len),dtype=int)
#         pos = np.zeros((end-ptr, batch_max_len),dtype=int)
#         y = np.zeros((1, end-ptr),dtype=int)
#         neg = np.zeros((1,n_negs, end-ptr),dtype=int)
#         for n in range(0, end-ptr):
#             seq[n, :len(training_data[ptr + n][0])] = training_data[ptr + n][0]
#             pos[n, :len(training_data[ptr + n][0])] = list(reversed(range(1,len(training_data[ptr + n][0])+1)))
#             seq_len.append(len(training_data[ptr + n][0]) - 1)
#         y[0,:]=[s[1] for s in training_data[ptr:end]]
#         for k in range(n_negs):
#             neg[0,k,:]=sample(item_list,end-ptr)
#         ptr=end
#         yield seq, pos, seq_len, y, neg

def next_batch_sequence(data, batch_size,n_negs=1,max_len=50):
    training_data = [item[1] for item in data.original_seq]
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    item_list = list(range(1,data.item_num+1))
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        y =np.zeros((batch_end-ptr, max_len),dtype=int)
        neg = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(training_data[ptr + n]) > max_len and -max_len or 0
            end =  len(training_data[ptr + n]) > max_len and max_len-1 or len(training_data[ptr + n])-1
            seq[n, :end] = training_data[ptr + n][start:-1]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
            y[n, :end]=training_data[ptr + n][start+1:]
            negatives=sample(item_list,end)
            while len(set(negatives).intersection(set(training_data[ptr + n][start:-1]))) >0:
                negatives = sample(item_list, end)
            neg[n,:end]=negatives
        ptr=batch_end
        yield seq, pos, y, neg, np.array(seq_len,int)

def next_batch_sequence_for_test(data, batch_size,max_len=50):
    sequences = [item[1] for item in data.original_seq]
    ptr = 0
    data_size = len(sequences)
    while ptr < data_size:
        if ptr+batch_size<data_size:
            batch_end = ptr+batch_size
        else:
            batch_end = data_size
        seq = np.zeros((batch_end-ptr, max_len),dtype=int)
        pos = np.zeros((batch_end-ptr, max_len),dtype=int)
        seq_len = []
        for n in range(0, batch_end-ptr):
            start = len(sequences[ptr + n]) > max_len and -max_len or 0
            end =  len(sequences[ptr + n]) > max_len and max_len or len(sequences[ptr + n])
            seq[n, :end] = sequences[ptr + n][start:]
            seq_len.append(end)
            pos[n, :end] = list(range(1,end+1))
        ptr=batch_end
        yield seq, pos, np.array(seq_len,int)