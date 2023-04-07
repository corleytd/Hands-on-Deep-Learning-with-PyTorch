# -*- coding: utf-8 -*-

'''
@Author   :   Corley Tang
@contact  :   cutercorleytd@gmail.com
@Github   :   https://github.com/corleytd
@Time     :   2023-03-23 12:37
@Project  :   Hands-on Deep Learning with PyTorch-linears
'''

import torch
from torch import nn
from torch.nn import functional as F


# 3层ReLU线性模型
class L3ReLULR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out


# 1层Sigmoid线性模型
class SigmoidLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        return out


# 2层Sigmoid线性模型
class L2SigmoidLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        return out


# 3层Sigmoid线性模型
class L3SigmoidLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        out = self.linear4(out)
        return out


# 4层Sigmoid线性模型
class L4SigmoidLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear5 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.sigmoid(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        out = self.linear3(out)
        out = torch.sigmoid(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)
        out = self.linear5(out)
        return out


# 2层Tanh线性模型
class L2TanhLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.tanh(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        out = self.linear3(out)
        return out


# 3层Tanh线性模型
class L3TanhLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.tanh(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        out = self.linear3(out)
        out = torch.tanh(out)
        out = self.linear4(out)
        return out


# 4层Tanh线性模型
class L4TanhLR(nn.Module):
    def __init__(self, in_features=2, hidden_dim=4, out_features=1, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.linear5 = nn.Linear(hidden_dim, out_features, bias=bias)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.tanh(out)
        out = self.linear2(out)
        out = torch.tanh(out)
        out = self.linear3(out)
        out = torch.tanh(out)
        out = self.linear4(out)
        out = torch.tanh(out)
        out = self.linear5(out)
        return out


# 2层激活函数带BN线性模型
class L2LRWithBN(nn.Module):
    def __init__(self, activation=F.relu, in_features=2, hidden_dim=4, out_features=1, bias=True, bn_mode=None,
                 bn_momentum=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear3 = nn.Linear(hidden_dim, out_features, bias=bias)
        self.activation = activation
        self.bn_mode = bn_mode

    def forward(self, x):
        if self.bn_mode == 'pre':  # 前置BN
            out = self.bn1(self.linear1(x))
            out = self.activation(out)
            out = self.bn2(self.linear2(out))
            out = self.activation(out)
            out = self.linear3(out)
        elif self.bn_mode == 'post':  # 后置BN
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(self.bn1(out))
            out = self.activation(out)
            out = self.linear3(self.bn2(out))
        else:  # 不使用BN层
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(out)
            out = self.activation(out)
            out = self.linear3(out)
        return out


# 3层激活函数带BN线性模型
class L3LRWithBN(nn.Module):
    def __init__(self, activation=F.relu, in_features=2, hidden_dim=4, out_features=1, bias=True, bn_mode=None,
                 bn_momentum=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear4 = nn.Linear(hidden_dim, out_features, bias=bias)
        self.activation = activation
        self.bn_mode = bn_mode

    def forward(self, x):
        if self.bn_mode == 'pre':  # 前置BN
            out = self.bn1(self.linear1(x))
            out = self.activation(out)
            out = self.bn2(self.linear2(out))
            out = self.activation(out)
            out = self.bn3(self.linear3(out))
            out = self.activation(out)
            out = self.linear4(out)
        elif self.bn_mode == 'post':  # 后置BN
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(self.bn1(out))
            out = self.activation(out)
            out = self.linear3(self.bn2(out))
            out = self.activation(out)
            out = self.linear4(self.bn3(out))
        else:  # 不使用BN层
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(out)
            out = self.activation(out)
            out = self.linear3(out)
            out = self.activation(out)
            out = self.linear4(out)
        return out


# 4层激活函数带BN线性模型
class L4LRWithBN(nn.Module):
    def __init__(self, activation=F.relu, in_features=2, hidden_dim=4, out_features=1, bias=True, bn_mode=None,
                 bn_momentum=0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_dim, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn3 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.bn4 = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        self.linear5 = nn.Linear(hidden_dim, out_features, bias=bias)
        self.activation = activation
        self.bn_mode = bn_mode

    def forward(self, x):
        if self.bn_mode == 'pre':  # 前置BN
            out = self.bn1(self.linear1(x))
            out = self.activation(out)
            out = self.bn2(self.linear2(out))
            out = self.activation(out)
            out = self.bn3(self.linear3(out))
            out = self.activation(out)
            out = self.bn4(self.linear4(out))
            out = self.activation(out)
            out = self.linear5(out)
        elif self.bn_mode == 'post':  # 后置BN
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(self.bn1(out))
            out = self.activation(out)
            out = self.linear3(self.bn2(out))
            out = self.activation(out)
            out = self.linear4(self.bn3(out))
            out = self.activation(out)
            out = self.linear5(self.bn4(out))
        else:  # 不使用BN层
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(out)
            out = self.activation(out)
            out = self.linear3(out)
            out = self.activation(out)
            out = self.linear4(out)
            out = self.activation(out)
            out = self.linear5(out)
        return out
