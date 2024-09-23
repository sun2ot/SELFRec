import yaml
from pathlib import Path

"""
合并 conda --from-history 和 --no-builds 导出的依赖项
"""

root = Path(__file__).parent.parent

with open(f'{root}/no_builds.env.yml', 'r') as file:
    # mamba env export --no-builds > environment.yml 
    no_builds = yaml.safe_load(file)

with open(f'{root}/history.env.yml', 'r') as file:
    # mamba env export --from-history  > history.env.yml
    history = yaml.safe_load(file)

# 合并两个文件
# 1. 保留 --from-history 的依赖
# 2. 添加 --no-builds 的版本号
no_builds_deps = {}
history_deps = {}
for item in no_builds['dependencies']:
    if isinstance(item, str):
        dep, version = item.split('=')
        no_builds_deps[dep] = version

for item in history['dependencies']:
    if isinstance(item, str):
        try:
            dep, version = item.split('=')
            history_deps[dep] = version
        except ValueError:
            history_deps[item] = ''

for dep, version in history_deps.items():
    if dep in no_builds_deps:
        history_deps[dep] = no_builds_deps[dep]

merged_dependencies = [f'{dep}={version}' for dep, version in history_deps.items()]

# 更新合并后的文件
history['dependencies'] = merged_dependencies
history['channels'] = no_builds['channels']
del history['prefix']

# 保存合并后的文件
with open(f'{root}/environment.yml', 'w') as file:
    yaml.dump(history, file)

