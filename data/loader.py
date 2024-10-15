import os.path
from os import remove
from re import split
from typing import Literal

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir: str, file: str, content, op='w'):
        """
        将内容写入指定路径和文件名的文件中

        Args:
            dir: 文件路径
            file: 文件名
            content: 写入的内容
            op: 写入方式，默认为`w`，即覆盖
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f"{dir}/{file}", op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file: str, rec_type: Literal['graph', 'sequential']):
        """
        根据模型类型加载数据集

        Args:
            file: 数据集路径
            rec_type: `graph` or `sequential`
        
        Returns:
            data (graph): List[[user, item, float(weight)], [...]]
        """
        if rec_type == 'graph':
            data = []
            with open(file) as f:
                for line in f:
                    # user_id, item_id, weight
                    items = line.strip().split(' ')
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    #? 为啥要转float???
                    # data.append([user_id, item_id, float(weight)])
                    data.append([user_id, item_id, weight])

        elif rec_type == 'sequential':
            data = {}
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    seq_id = items[0]
                    data[seq_id]=items[1].split()
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data
    
