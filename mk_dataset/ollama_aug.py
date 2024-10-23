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
from safetensors.torch import load_file

from tqdm import tqdm
import requests
from util.logger import Log

parser = argparse.ArgumentParser(description='LLM augmentation configurations.')
parser.add_argument('--type', choices=['normal', 'specific'], default='normal', help='augmentation type: normal or specific')
parser.add_argument('--skip', type=int, default=0, help='Skip the augmentation process based on item index.')
parser.add_argument('--text', type=str, default='yelp_out/yelp_text.json', help='text file path.')
parser.add_argument('--interact', type=str, default='yelp_ds/filter_yelp_interactions.txt', help='interaction dataset path.')
parser.add_argument('--template', type=str, default='aug_prompt.txt', help='LLM augmentation prompt template.')
parser.add_argument('--model', type=str, default="qwen2.5:0.5b", help='LLM model.')
parser.add_argument('--dsname', type=str, default="yelp", help='output file prefix name.')
parser.add_argument('--output', type=str, default='yelp_out', help='output directory path.')
args = parser.parse_args()

"""配置"""
# ollama
host = "http://127.0.0.1:11434/api/generate"

# 提示词模板
with open(args.template, 'r') as f:
    template = f.read()

# 加载文本数据
with open(args.text, 'r') as text_file:
    item_text = json.load(text_file)

def get_response(url, data):
    """ollama 请求函数"""
    response = requests.post(url, json=data, timeout=20)
    response_dict = json.loads(response.text)
    response_content = response_dict["response"]
    return response_content

# 初始化日志
now = datetime.now().strftime('%Y%m%d_%H%M')
llm_log = Log(module='llm_aug', filename=f'llm_aug_{now}')


#* 初始化用户历史交互记录
# {
#     'user': {
#         'item1': 'categories1',
#         'item2': 'categories2',
#         ...
#     }
# }
user_history: dict[str, dict[str, str]] = {}

#* 全局交互集 user -> set(items)
interactions: dict[str, set] = defaultdict(set)
with open(args.interact, 'r') as file:
    for line in file:
        user, item = line.split(' ')[0], line.split(' ')[1]
        interactions[user].add(item)

# 构建用户历史交互数据
if os.path.exists(f"{args.output}/{args.dsname}_user_history.json"):
    llm_log.add('File exists. Skipping user history construction!')
    with open(f"{args.output}/{args.dsname}_user_history.json", 'r') as file:
        user_history = json.load(file)
else:
    print("'construct user history'")
    for idx, user in enumerate(interactions):
        items = list(interactions[user])
        if len(items) == 0:
            raise ValueError(f'{user} get no items')
        # 为保证模型输出效果&效率, 限制上下文数量
        selected = random.sample(items, min(3, len(items)))
        his_dict: dict[str, str] = {item: item_text[item] for item in selected}
        user_history[user] = his_dict

    #* 持久化历史交互数据
    with open(f"{args.output}/{args.dsname}_user_history.json", 'w', encoding='utf-8') as file:
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

    #* mode1: 处理完成后补全缺失(生成失败)数据
    error_num = 0
    error_users: list[str] = []
    with open('ifashion_out/ifashion_user_pref.json', 'r') as file:
        #* 从合并后的文件中找出存在缺失的 user
        pre_data: dict[str, str] = json.load(file)
        for k, v in pre_data.items():
            if v.strip() == "":
                error_num += 1
                error_users.append(k)
        print(f"total {len(pre_data)} users, {error_num} empty fields")
    user_history = {user: user_history[user] for user in error_users}

    #* mode2: 在已生成的结果中补充缺少的用户
    # user_pref_embs = load_file("/home/yzh/code/SELFRec/mk_dataset/yelp_ds/user_pre_embs.safetensors")
    # remain_user = set()
    # for user in interactions:
    #     if user in user_pref_embs: continue
    #     remain_user.add(user)
    # print(f"remain: {len(remain_user)}")
    # user_history = {user: user_history[user] for user in remain_user}

else:
    raise ValueError('type error')

for user, his_dict in tqdm(user_history.items(), desc='llm augment'):
    try:
        # 构建 {history} 提示词
        history_list = [f'{item}: {categories}' for item, categories in his_dict.items()]
        form_history = '\n'.join(history_list)
        prompt = template.format(history=form_history)

        # 调用模型
        data = {
            "model": args.model,
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
        with open(f'{args.output}/{args.dsname}_user_preference-{now}.json', 'a', encoding='utf-8') as file:
            for user, pref in user_preference.items():
                file.write(json.dumps({user: pref}, ensure_ascii=False) + '\n')
        user_preference.clear()  # 清空缓存


if len(user_preference) > 0: # 写入剩余内容
    with open(f'{args.output}/{args.dsname}_user_preference-{now}.json', 'a', encoding='utf-8') as file:
        for user, pref in user_preference.items():
            file.write(json.dumps({user: pref}, ensure_ascii=False) + '\n')

print('finish')
llm_log.add('finish')