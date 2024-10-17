from SELFRec import SELFRec
from util.conf import ModelConf
import argparse

# 因为实验环境一些令人无语的历史原因, 这里添加一个配置文件的覆盖接口
parser = argparse.ArgumentParser(description="Used to replace the configuration file path.")
parser.add_argument('--config', '-c', type=str, default=None, help='Path to the configuration file.')
args = parser.parse_args()

def print_models(title, models):
    print(f"{'=' * 80}\n{title}\n{'-' * 80}")
    for category, model_list in models.items():
        print(f"{category}:\n   {'   '.join(model_list)}\n{'-' * 80}")

if __name__ == '__main__':
    models = {
        'Graph-Based Baseline Models': ['LightGCN', 'DirectAU', 'MF'],
        'Self-Supervised Graph-Based Models': ['SGL', 'SimGCL', 'SEPT', 'MHCN', 'BUIR', 'SelfCF', 'SSL4Rec', 'XSimGCL', 'NCL', 'MixGCF'],
        'Sequential Baseline Models': ['SASRec'],
        'Self-Supervised Sequential Models': ['CL4SRec', 'BERT4Rec']
    }

    # 输出 CLI 交互菜单
    print('=' * 80)
    print('   SELFRec: A library for self-supervised recommendation.   ')
    print_models("Available Models", models)

    model = input('Please enter the model you want to run:')

    all_models = sum(models.values(), [])
    if model in all_models:
        if args.config is not None:
            conf = ModelConf(args.config)
        else:
            conf = ModelConf(f'./conf/{model}.yaml')
        rec = SELFRec(conf)
        rec.execute()
    else:
        print('Wrong model name!')
        exit(-1)
