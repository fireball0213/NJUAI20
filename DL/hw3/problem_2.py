# -*- coding: UTF-8 -*- #
"""
@filename:problem_2.py
@author:201300086
@time:2022-12-17
"""

import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

# 卷积网络
net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    )
# 全连接网络
classifier = nn.Sequential(nn.Linear(in_features=144, out_features=3, bias=True), )

# 查看网络信息
summary(net, (3, 128, 128))
summary(classifier, (64, 144))

# 测试
x = torch.rand(3, 128, 128)
x = net(x)
x = torch.flatten(x, start_dim=1)  # 从维度1开始展平
x = classifier(x)
