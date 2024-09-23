import logging
import os


class Log(object):
    _instance = None
    _filename = None
    _module = None

    # 单例模式
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Log, cls).__new__(cls)
        return cls._instance

    def __init__(self, module: str = None, filename: str = None):
        """
        初始化日志模块(单例)self.logger并添加FileHandler

        Args:
            module: 模型名称 -> 日志模块名
            filename: 日志文件名(存储路径)
        """
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(module)
            self.logger.setLevel(level=logging.INFO)
            os.makedirs('./log/', exist_ok=True)

            try:
                # 仅在第一次创建时生成 filename
                if Log._filename is None and Log._module is None:
                    if module and filename:
                        Log._filename = filename
                        Log._module = module
                    else:
                        raise ValueError('module and filename are required')
                else:
                    if filename or module: 
                        raise ValueError('Log is a singleton, module/filename is not allowed to be changed')
            except ValueError as e:
                print(e)

            handler = logging.FileHandler(f'./log/{Log._filename}.log')
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
