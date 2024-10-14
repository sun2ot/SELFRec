import os
import yaml


class ModelConf(object):
    def __init__(self, file: str):
        """配置文件类

        Args:
            file (str): yaml 配置文件路径
        """
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item: str):
        """获取配置项"""
        if not self.contain(item):
            raise KeyError(f"Parameter {item} not found in the configuration file!")
        return self.config[item]

    def contain(self, key: str) -> bool:
        return key in self.config

    def read_configuration(self, file: str) -> None:
        if not os.path.exists(file):
            raise IOError("Config file is not found!")
        with open(file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error in configuration file: {exc}")
                raise IOError
