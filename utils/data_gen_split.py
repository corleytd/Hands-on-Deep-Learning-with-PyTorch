# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-01-27 23:35
@Project  :   Hands-on Deep Learning with PyTorch-data_gen_split
数据的生成与切分
'''

import random

import torch
from torch.utils.data import random_split, DataLoader


def gen_reg_data(num_samples=1000, w=(1, 1, 0), deg=1, delta=0.01, bias=True):
    '''
    创建回归类数据集
    :param num_samples: 数据集样本量
    :param w: 特征系数向量（包含截距）
    :param deg: 多项式关系的最高次项
    :param delta: 扰动项系数
    :param bias: 是否需要截距
    :return: 生成的特征张量和标签
    '''
    if bias:
        num_features = len(w) - 1  # 特征数
        w_true = torch.tensor(w[:-1], dtype=torch.float)  # 特征系数
        b_true = torch.tensor(w[-1], dtype=torch.float)  # 截距
        features_true = torch.randn(num_samples, num_features)  # 特征张量
        if num_features == 1:  # 若输入特征只有1个，则不能使用矩阵乘法
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            labels_true = torch.mv(features_true.pow(deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(features_true.shape[0], 1)), 1)  # 在特征张量的最后添加1列1
        labels = labels_true + torch.randn_like(labels_true) * delta
    else:
        num_features = len(w)
        w_true = torch.tensor(w, dtype=torch.float)
        features = torch.randn(num_samples, num_features)
        if num_features == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mv(features.pow(deg), w_true)
        labels = labels_true + torch.randn_like(labels_true) * delta
    return features, labels


def gen_cls_data(num_features=2, num_samples=500, num_classes=2, dispersions=(4, 2), bias=False):
    '''
    生成分类数据
    :param num_features: 特征数量
    :param num_samples: 每个类别样本数量
    :param num_classes: 类别数
    :param dispersions: 数据分布离散程度，表示(数组均值, 数组标准差)
    :param bias: 建立回归模型时是否需要带入截距
    :return: 特征张量和标签，分别是浮点型和长整型
    '''
    label_ = torch.empty(num_samples, 1)  # 每一类标签的参考
    mean_, std_ = dispersions  # 每一类特征张量的均值和方差
    features, labels = [], []  # 存储每一类别的特征张量和标签
    k = mean_ * (num_classes - 1) / 2  # 每一类特征张量均值的惩罚因子，实现对分布离散程度的控制

    for i in range(num_classes):
        cur_features = torch.normal(i * mean_ - k, std_, size=(num_samples, num_features))  # 每一类特征张量
        cur_labels = torch.full_like(label_, i)  # 每一类标签
        features.append(cur_features)
        labels.append(cur_labels)

    # 合并数据
    features = torch.cat(features).float()
    labels = torch.cat(labels).long()

    # 有截距
    if bias:
        features = torch.cat((features, torch.ones(features.size(0), 1)), -1)

    return features, labels


def data_split_batches(batch_size, features, labels, shuffle=True):
    '''
    数据切分
    :param batch_size: 分配的大小，即每个小批量包含多少样本
    :param features: 特征张量
    :param labels: 标签
    :param shuffle: 是否随机打乱
    :return: 由切分后的特征和标签组成的列表
    '''
    num_samples = labels.size(0)
    indices = list(range(num_samples))
    if shuffle:  # 随机打乱
        random.shuffle(indices)
    batches = []
    for idx in range(0, num_samples, batch_size):
        slice_indices = torch.tensor(indices[idx:idx + batch_size])
        batches.append((torch.index_select(features, 0, slice_indices), torch.index_select(labels, 0, slice_indices)))

    return batches


def split_load_data(dataset_class, features, labels, batch_size=16, train_ratio=0.8):
    '''
    数据封装、切分和加载
    :param features: 特征
    :param labels: 标签
    :param batch_size: 批大小
    :param train_ratio: 训练集数据占比
    :return: 加载好的训练集和测试集
    '''
    data = dataset_class(features, labels)
    num_samples = len(features)
    num_train = int(num_samples * train_ratio)
    num_test = num_samples - num_train
    train_data, test_data = random_split(data, [num_train, num_test])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size * 2)
    return train_loader, test_loader
