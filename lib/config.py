import yaml
import torch

import lib.models as models
import lib.datasets as datasets


class Config(object):
    def __init__(self, config_path):
        # 加载 ymal 文件
        self.config = {}
        self.load(config_path)

    def load(self, path):
        # 加载 ymal 文件
        with open(path, 'r') as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        return self.config_str

    def get_dataset(self, split):
        # 获取指定数据集
        return getattr(datasets,
                       self.config['datasets'][split]['type'])(**self.config['datasets'][split]['parameters'])

    def get_model(self):
         # 获取模型
        name = self.config['model']['name']
        parameters = self.config['model']['parameters']
        return getattr(models, name)(**parameters)

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_lr_scheduler(self, optimizer):
        # 获取学习率调度器
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def get_loss_parameters(self):
        # 获取loss参数
        return self.config['loss_parameters']

    def get_test_parameters(self):
        return self.config['test_parameters']

    def __getitem__(self, item):
        return self.config[item]
