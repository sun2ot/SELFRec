import os.path
from os import remove
from re import split
from typing import Literal
import torch
import torch.nn as nn
from safetensors import safe_open
from util.logger import Log
import tqdm
import json

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
        with open(dir + file, op) as f:
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
    

    @staticmethod
    def load_image_data(image_set: str, 
                        item2image: str, 
                        emb_size: int, 
                        device_id: int) -> dict[str, torch.Tensor]:
        """
        加载图片数据集：
        读取图像预处理数据并从CLIP模型输出的512维投影到embedding_size

        Args:
            image_set (str): safetensors 文件路径
            item2image (str): item -> 图片映射文件
            emb_size (int): 投影维度
        
        Returns:
            image_embs (dict): item (str) -> mean image embedding (pytorch tensor)
        """
        image_embs: dict[str, torch.Tensor] = {}  # item -> mean image embedding
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        # 定义一个线性层将512维图像特征映射到emb_size
        linear_projection = nn.Linear(512, emb_size, device=device)

        with safe_open(image_set, 'pt', device=f"cuda:{device_id}") as image_safetensors: # type: ignore
            with open(item2image, 'r') as map_file:
                Log.cli('Loader', f'📷 Loading image safetensors to cuda:{device_id} and project to {emb_size} dimensions')
                for line in tqdm.tqdm(map_file, desc='items'):
                    item = line.strip().split(' ')[0]
                    images = line.strip().split(' ')[1:]
                    try:
                        image_embs[item] = linear_projection(
                            torch.mean(
                            torch.stack([image_safetensors.get_tensor(image) for image in images]), dim=0)
                            )
                    except Exception as e:
                        Log.catch(e, item, 'item2photo emb project')
                        # print(f'\n{"-"*50}\n{item} error:\n{e}\n{"-"*50}')
                        exit(-1)
                        
        return image_embs


    @staticmethod
    def load_text_data(path: str, device_id: int) -> safe_open:
        """加载文本模态数据: 读取文本预处理数据并从stella模型输出的1024维投影到embedding_size

        Args:
            path (str): safetensors 文件路径

        Returns:
            text_embs (dict[str, torch.Tensor])
        """
        #! 这里跟图像模态做个区分尝试, 直接返回一个 safe_open class, 因为这玩意是 lazy load
        # 正好统一逻辑, 数据处理在 Interaction 里做
        # 所以某种意义上说, safetensors 官方的 docs 是错的... 因为 safe_open 没有实现 with 上下文功能
        text_safetensors =  safe_open(path, 'pt', device=f"cuda:{device_id}")
        Log.cli('Loader', f'📒 Loading text safetensors to cuda:{device_id}')
        
        return text_safetensors