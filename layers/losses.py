# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-03-21 18:20
@Project  :   Hands-on Deep Learning with PyTorch-losses
'''


def loss_with_loader(model, data_loader, criterion):
    '''
    对整个数据集计算损失
    :param model: 模型
    :param data_loader: 加载好的数据
    :param criterion: 损失函数
    :return: 损失值
    '''
    data = data_loader.dataset
    X = data[:][0]
    y = data[:][1]
    y_hat = model(X)
    return criterion(y_hat, y)