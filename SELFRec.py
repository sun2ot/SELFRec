from data.loader import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config

        print('Reading data and preprocessing...')

        self.training_data = FileIO.load_data_set(config['training.set'], config['model']['type'])
        self.test_data = FileIO.load_data_set(config['test.set'], config['model']['type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        if config.contain('image_modal'):
            self.kwargs['image_embs'] = FileIO.load_image_data(
                config['image_modal']['image_set'],
                config['image_modal']['item2image'],
                config['embedding.size']
            )


    def execute(self):
        # import the model module
        import_str = f"from model.{self.config['model']['type']}.{self.config['model']['name']} import {self.config['model']['name']}"
        # from model.graph.XSimGCL import XSimGCL
        exec(import_str)
        recommender = f"{self.config['model']['name']}(self.config,self.training_data,self.test_data,**self.kwargs)"
        # XSimGCL(self.config,self.training_data,self.test_data,**self.kwargs)
        eval(recommender).execute()
