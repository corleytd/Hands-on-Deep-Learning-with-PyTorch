# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-03-21 18:10
@Project  :   Hands-on Deep Learning with PyTorch-train_utils
'''

import torch
from torch import nn
from torch import optim
from tqdm import tqdm

from layers.losses import loss_with_loader


def fit_one_epoch(model, criterion, optimizer, data_loader, cla=False):
    '''
    模型训练一轮
    :param model: 待训练的模型
    :param criterion: 损失函数
    :param optimizer: 优化算法
    :param data_loader: 数据
    :param cla: 是否是分类任务
    :return: None
    '''
    for X, y in data_loader:
        optimizer.zero_grad()
        y_hat = model(X)
        if cla:
            y = y.int()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()


def fit(model, criterion, optimizer, data_loader, cla=False, num_epochs=5):
    '''
    模型训练
    :param model: 待训练的模型
    :param criterion: 损失函数
    :param optimizer: 优化算法
    :param data_loader: 数据
    :param cla: 是否是分类任务
    :param num_epochs: 训练轮数
    :return: None
    '''
    for _ in tqdm(range(num_epochs), desc='Epoch'):
        fit_one_epoch(model, criterion, optimizer, data_loader, cla)


def train_test_model(model, train_loader, test_loader, criterion=nn.MSELoss, optimizer_class=optim.SGD,
                     evaluation=loss_with_loader, lr=0.03, num_epochs=20, cla=False):
    '''
    训练，并记录训练过程中训练集和测试集上的损失变化
    :param model: 待训练的模型
    :param train_loader: 训练数据集
    :param test_loader: 测试数据集
    :param criterion: 损失函数
    :param optimizer_class: 优化器
    :param evaluation: 数据集损失计算函数
    :param lr: 学习率
    :param num_epochs: 训练轮数
    :param cla: 是否是分类任务
    :return: 训练损失和测试损失
    '''
    criterion = criterion()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for _ in tqdm(range(num_epochs), desc='Epoch'):
        model.train()
        fit_one_epoch(model, criterion, optimizer, train_loader, cla)
        model.eval()
        train_losses.append(evaluation(model, train_loader, criterion).item())
        test_losses.append(evaluation(model, test_loader, criterion).item())
    return train_losses, test_losses


def compare_models(model_list, train_loader, test_loader, criterion=nn.MSELoss, optimizer_class=optim.SGD,
                   evaluation=loss_with_loader, lr=0.03, num_epochs=20, cla=False):
    '''
    模型对比
    :param model_list: 模型对象列表
    :param train_loader: 训练集
    :param test_loader: 测试集
    :param criterion: 损失函数
    :param optimizer_class: 优化器
    :param evaluation: 数据集损失计算函数
    :param lr: 学习率
    :param num_epochs: 训练轮数
    :param cla: 是否是分类任务
    :return: 训练损失和测试损失
    '''
    criterion = criterion()

    train_losses, test_losses = torch.zeros(len(model_list), num_epochs), torch.zeros(len(model_list), num_epochs)

    # 训练模型
    for epoch in tqdm(range(num_epochs), desc='Epoch'):
        for idx, model in enumerate(model_list):
            model.train()
            optimizer = optimizer_class(model.parameters(), lr=lr)
            fit_one_epoch(model, criterion, optimizer, train_loader, cla=cla)
            model.eval()
            train_losses[idx, epoch] = evaluation(model, train_loader, criterion).item()
            test_losses[idx, epoch] = evaluation(model, test_loader, criterion).item()

    return train_losses, test_losses
