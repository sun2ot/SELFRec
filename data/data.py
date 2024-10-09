class Data(object):
    """模型输入数据封装(配置, 训练集, 测试集)"""
    def __init__(self, conf, training: list[list[str]], test):
        self.config = conf
        self.training_data = training
        # 测试集跟验证集通用
        self.test_data = test #can also be validation set if the input is for validation







