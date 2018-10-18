import torch
import torch.nn.functional as F
from torch import autograd,nn
import numpy as nn

import matplotlib.pyplot as plt
from torchvision import transforms,datasets

#data format NCHW samples, channels height width

input_image = autograd.Variable(torch.randn(1,3,32,32))
print(input_image.shape)

#some RGB but some is BGR  or ARGB
#1D filter (out_channels,in_channels time)
input_signal = autograd.Variable(torch.randn(1,40,100)) # 40 dimensional signal for 100 timesteps
print(input_signal.size())

#torch.nn提供cnn
"""
torch.nn.Conv2d
torch.nn.ConvTranspose2d
torch.nn.MaxPool2d
torch.nn.AvgPool2d
"""

#创建layer,stride跳取，padding 拓展
layer_c2d = torch.nn.Conv2d(in_channels=3,out_channels=20,kernel_size=5,stride=1,padding=2)
layer_avg = torch.nn.AvgPool2d(kernel_size=32)

#使用layer
y = layer_c2d(input_image)
print(y.size())

#将layer 添加到模型中
model = torch.nn.Sequential(layer_avg,layer_c2d)
y = model(input_image)
print(y.size())

#卷积函数
#torch.nn.functional
#F.max_pool2d，F.dropout2d, F.cov2d

filters= autograd.variable(torch.randn(20,3,5,5))# 5x5 filter from 3 dimensions to 20


