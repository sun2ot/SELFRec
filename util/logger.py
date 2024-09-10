import logging
import os


class Log(object):
    def __init__(self, module: str, filename: str):
        """
        Args:
            module: 模型名称 -> 日志模块名
            filename: 日志文件名(存储路径)
        """
        self.logger = logging.getLogger(module)
        self.logger.setLevel(level=logging.INFO)
        os.makedirs('./log/', exist_ok=True)
        handler = logging.FileHandler('./log/'+filename+'.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def add(self,text):
        """输出日志内容"""
        self.logger.info(text)
    
    @staticmethod
    def catch(exception: Exception, subject: str, position: str = ''):
        """CLI输出自定义异常捕获日志"""
        print(f'\n{"-"*50}\n{position} {subject} error:\n{exception}\n{"-"*50}')
