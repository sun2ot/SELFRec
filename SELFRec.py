from data.loader import FileIO
from time import strftime, localtime, time

class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config

        print('Reading data and preprocessing...')

        self.training_data = FileIO.load_data_set(config['training.set'], config['model']['type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model']['type'])

        self.kwargs = {}
        # 传入时间戳, 用于文件持久化
        self.kwargs['timestamp'] = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.kwargs['model_name'] = config['model']['name']

        if config.contain('social.data'):
            social_data = FileIO.load_social_data(config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        #* 图像模态
        if config.contain('image_modal') and config['image_modal']['fusion']:
            self.kwargs['image_modal'] = config['image_modal']
        
        #* 文本模态
        if config.contain('text_modal') and config['text_modal']['fusion']:
            self.kwargs['text_modal'] = config['text_modal']

    def execute(self):
        # import the model module
        import_str = f"from model.{self.config['model']['type']}.{self.config['model']['name']} import {self.config['model']['name']}"
        # from model.graph.XSimGCL import XSimGCL
        exec(import_str)
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        # XSimGCL(self.config,self.training_data,self.test_data,**self.kwargs)
        eval(recommender).execute()
