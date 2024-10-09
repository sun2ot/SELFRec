import sys
import os
from os import path
# 将父目录添加到sys.path中
root_dir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, root_dir)
import json
import time
import random
random.seed(666)
from collections import defaultdict
from datetime import datetime
import argparse

from tqdm import tqdm
import requests

from data.loader import FileIO
from util.conf import ModelConf
from util.logger import Log

parser = argparse.ArgumentParser(description='LLM augmentation configurations.')
parser.add_argument('--type', choices=['normal', 'specific'], default='normal', help='augmentation type: normal or specific')
parser.add_argument('--skip', type=int, default=0, help='Skip the augmentation process based on item index.')
parser.add_argument('--input', type=str, default='yelp_out/yelp_text.json', help='text file path.')
parser.add_argument('--output', type=str, default='yelp_out', help='output directory path.')
args = parser.parse_args()

# 加载配置文件
config = ModelConf('/home/yzh/code/SELFRec/conf/XSimGCL.yaml')
host = "http://127.0.0.1:11434/api/generate"


def get_response(url, data):
    """ollama 请求函数"""
    response = requests.post(url, json=data, timeout=20)
    response_dict = json.loads(response.text)
    response_content = response_dict["response"]
    return response_content

# 初始化日志
now = datetime.now().strftime('%Y%m%d_%H%M')
llm_log = Log(module='llm_aug', filename=f'llm_aug_{now}')

# 加载提示词模板
template_file = '/home/yzh/code/SELFRec/conf/aug_prompt.txt'
with open(template_file, 'r') as f:
    template = f.read()

# 加载训练集
training_data = FileIO.load_data_set('/home/yzh/code/SELFRec/mk_dataset/yelp_ds_final/train_data.txt',
                                     config['model']['type'])
# 加载文本数据
with open(args.input, 'r') as text_file:
    yelp_text = json.load(text_file)

# 初始化双向映射集合
training_set_u: dict[str, dict[str, str]] = defaultdict(dict)
for user, item, rating in training_data:
    training_set_u[user][item] = rating

# 初始化用户历史交互记录
# {
#     'user': {
#         'item1': 'categories1',
#         'item2': 'categories2',
#         ...
#     }
# }
user_history: dict[str, dict[str, str]] = {}

#* 训练集用户
users = training_set_u.keys()

# 构建用户历史交互数据
#* sha256: 7fab614c1d970735e7d6e328943ba645ea5c53a2a59ba9aec78667ae256f326e
if os.path.exists(f"{args.output}/yelp_user_history.json"):
    llm_log.add('File exists. Skipping user history construction!')
    with open(f"{args.output}/yelp_user_history.json", 'r') as file:
        user_history = json.load(file)
else:
    for idx, user in tqdm(enumerate(users), desc='construct user history'):

        items = list(training_set_u.get(user, {}).keys())
        if len(items) == 0:
            raise ValueError(f'{user} get no items')
        # 为保证模型输出效果&效率, 限制上下文数量
        selected = random.sample(items, min(3, len(items)))
        his: dict[str, str] = {item: yelp_text[item] for item in selected}
        user_history[user] = his

    #* 持久化历史交互数据
    with open(f"{args.output}/yelp_user_history.json", 'w', encoding='utf-8') as file:
        json.dump(user_history, file, ensure_ascii=False)


#* 开始数据增强
llm_log.add('llm augmentation start...')

# 初始化用户偏好
#! 为了让代码执行过程可中继，持久化时将采用 json line 的形式，以避免生成单一对象再写入文件
user_preference: dict[str, str] = {}

if args.type == 'normal':
    # 为支持代码中继运行, 添加 skip 参数
    user_history = dict(list(user_history.items())[args.skip:])
elif args.type == 'specific':
    """Notice

    1. specific 模式用于仅对特定用户进行数据增强
    2. 此处实现逻辑不唯一, 只要能提供需要 specific 的 user_list 即可
    3. 为了不破坏原始增强逻辑, 这里不对原文件进行修改, 而是同样生成新文件
    """
    error_num = 0
    error_users: list[str] = []
    with open('yelp_user_preferences.v1.json', 'r') as file:
        #* 从合并后的文件中找出存在缺失的 user
        pre_data: dict[str, str] = json.load(file)
        for k, v in pre_data.items():
            if v.strip() == "":
                error_num += 1
                error_users.append(k)
        print(f"total {len(pre_data)} users, {error_num} empty fields")
    user_history = {user: user_history[user] for user in error_users}
else:
    raise ValueError('type error')

for user, his in tqdm(user_history.items(), desc='llm augment'):
    try:
        # 构建 {history} 提示词
        history_list = [f'{item}: {categories}' for item, categories in his.items()]
        form_history = '\n'.join(history_list)
        prompt = template.format(history=form_history)

        # 调用模型
        data = {
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        }
        res = get_response(host, data)
        user_preference[user] = res
    except Exception as e:
        llm_log.add(f'{user} error: {e}')
        user_preference[user] = ''
    
    if len(user_preference) % 100 == 0 and len(user_preference) > 0:
        print(f'processed {len(user_preference)}')
        #* 持久化用户偏好数据
        with open(f'{args.output}/yelp_user_preference-{now}.json', 'a', encoding='utf-8') as file:
            for user, pref in user_preference.items():
                file.write(json.dumps({user: pref}, ensure_ascii=False) + '\n')
        user_preference = {}

#* 写入剩余内容
with open(f'{args.output}/yelp_user_preference-{now}.json', 'a', encoding='utf-8') as file:
    for user, pref in user_preference.items():
        file.write(json.dumps({user: pref}, ensure_ascii=False) + '\n')
        
print('finish')