# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-03-25 11:34
@Project  :   Hands-on Deep Learning with PyTorch-visualization_utils
'''

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from torch import nn


def violin_plot_layers(model, is_grad=True):
    '''
    绘制模型参数或梯度的小提琴图
    :param model: 模型
    :param is_grad: 是否画梯度
    :return: None
    '''
    layer_params = []

    # 记录梯度
    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Linear):
            if is_grad:
                layer_param = module.weight.grad.reshape(-1, 1).numpy()  # 每一层的梯度
            else:
                layer_param = module.weight.detach().reshape(-1, 1).numpy()  # 每一层的权重
            index = np.full_like(layer_param, idx)  # 对层进行标号
            layer = np.concatenate((layer_param, index), -1)
            layer_params.append(layer)

    # 拼接各层
    layer_params = np.concatenate(layer_params, 0)

    # 绘制图像
    ax = sns.violinplot(y=layer_params[:, 0], x=layer_params[:, 1])
    ax.set(xlabel='layer', title='grad' if is_grad else 'weight')
    plt.show()
