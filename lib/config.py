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
        # datasets:
        #   train:
        #     type: # LaneDataset
        #     parameters:
        #       dataset: tusimple
        #       split: train
        #       img_size: [360, 640]
        #       normalize: true
        #       aug_chance: 0.9090909090909091 # 10/11
        #       augmentations: # ImgAug augmentations
        #        - name: Affine
        #          parameters:
        #            rotate: !!python/tuple [-10, 10]
        #        - name: HorizontalFlip
        #          parameters:
        #            p: 0.5
        #        - name: CropToFixedSize
        #          parameters:
        #            width: 1152
        #            height: 648
        #       root: "../dataset/TUSimple/train_set/" # Dataset root
        #
        #   test: &test
        #     type: # LaneDataset
        #     parameters:
        #       dataset: tusimple
        #       split: test
        #       img_size: [360, 640]
        #       root: "../dataset/TUSimple/test_set/"
        #       normalize: true
        #       augmentations: []
        #
        return getattr(datasets,
                       self.config['datasets'][split]['type'])(**self.config['datasets'][split]['parameters'])

    def get_model(self):
        # 获取模型
        # model:
        # name: PolyRegression
        # parameters:
        #   num_outputs: 35 # (5 lanes) * (1 conf + 2 (upper & lower) + 4 poly coeffs)
        #   pretrained: true
        #   backbone: 'efficientnet-b0'
        #   pred_category: false
        name = self.config['model']['name']
        parameters = self.config['model']['parameters']
        return getattr(models, name)(**parameters)

    def get_optimizer(self, model_parameters):
        # optimizer:
        #   name: Adam
        #   parameters:
        #     lr: 3.0e-4
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_lr_scheduler(self, optimizer):
        # 获取学习率调度器
        # lr_scheduler:
        #   name: CosineAnnealingLR
        #   parameters:
        #       T_max: 385
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def get_loss_parameters(self):
        # 获取loss参数
        # loss_parameters:
        #   conf_weight: 1
        #   lower_weight: 1
        #   upper_weight: 1
        #   cls_weight: 0
        #   poly_weight: 300
        return self.config['loss_parameters']

    def get_test_parameters(self):
        # test_parameters:
        #   conf_threshold: 0.5 # Set predictions with confidence lower than this to 0 (i.e., set as invalid for the metrics)
        return self.config['test_parameters']

    def __getitem__(self, item):
        return self.config[item]
