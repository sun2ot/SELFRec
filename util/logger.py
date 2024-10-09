from __future__ import annotations
import logging
import os


class Log(object):
    _instances: dict[str, Log] = {}  # 存储已创建的Log实例

    #* 单例模式
    def __new__(cls, module, filename: str=''):
        # 模块名作为键查找已存在的Log实例
        if module not in cls._instances:
            cls._instances[module] = super(Log, cls).__new__(cls)
        return cls._instances[module]

    def __init__(self, *, module: str, filename: str=''):
        """
        为module创建logger (DEBUG) 并添加FileHandler

        Args:
            module: 模块名称 -> 日志模块名
            filename: 日志文件名(存储路径)
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(module)
            self.logger.setLevel(logging.DEBUG)
            os.makedirs('./log/', exist_ok=True)
            handler = logging.FileHandler(f'./log/{filename}.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def addFileHandler(self, module: str, filename: str, level): # 以防万一写着，但是最好别用
        """为module对应的logger添加FileHandler

        Args:
            module (str): 模块名称(单例)
            filename (str): 文件名(无后缀)
            level (logging.__all__): 日志级别(>DEBUG)
        """
        log_instance = self._instances[module]
        handler = logging.FileHandler(f'./log/{filename}.log')
        handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        log_instance.logger.addHandler(handler)

    def add(self,text):
        """写入info日志"""
        self.logger.info(text)

    def error(self, text):
        self.logger.error(text)

    def warning(self, text):
        self.logger.warning(text)
    
    @staticmethod
    def catch(exception: Exception, subject: str, position: str = '') -> None:
        """CLI输出自定义异常捕获日志"""
        print(f'\n{"-"*50}\n{position} {subject} error:\n{exception}\n{"-"*50}')
    
    @staticmethod
    def cli(positon: str, content: str='') -> None:
        print(f'[{positon}]: {content}')

    @staticmethod
    def raiseErr(positon: str, content: str='') -> None:
        exception = Exception(f'[{positon}]: {content}')
        raise exception