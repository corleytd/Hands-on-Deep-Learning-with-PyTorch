# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-03-21 18:01
@Project  :   Hands-on Deep Learning with PyTorch-datasets
'''

from torch.utils.data import Dataset


class GenDataset(Dataset):
    '''自定义生成数据Dataset'''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]