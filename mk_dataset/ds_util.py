import json

def head(path: str, num: int = 1, jsonf: bool = False):
    """
    读取json文件的前几行

    Args:
        path: 文件路径
        num: 读取的行数
        jsonf: 是否以json格式输出
    """
    with open(path, 'r', encoding='utf-8') as f:
        for _ in range(num):
            if jsonf:
                json_data = json.loads(next(f))
                json_str = json.dumps(json_data ,ensure_ascii=True, indent=2)
                print(json_str)
            else: print(next(f))