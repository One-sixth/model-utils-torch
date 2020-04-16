'''
模板类和函数

一般情况是从本模块中复制所需的模块到自己的项目中，然后修改
当然也可直接使用，不过不推荐
'''
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearBnAct(nn.Module):
    def __init__(self, in_feat, out_feat, act, eps=1e-8, mom=0.9):
        super().__init__()
        self.lin = nn.Linear(in_feat, out_feat, bias=False)
        self.norm = nn.BatchNorm1d(out_feat, eps=eps, momentum=mom)
        self.act = act

    def forward(self, x):
        y = x
        y = self.lin(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class ConvBnAct2D(nn.Module):
    def __init__(self, in_ch, out_ch, ker_sz, stride, pad, act, dila=1, groups=1, eps=1e-8, mom=0.9):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, dilation=dila, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=eps, momentum=mom)
        self.act = act

    def forward(self, x):
        y = x
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class DeConvBnAct2D(nn.Module):
    def __init__(self, in_ch, out_ch, ker_sz, stride, pad, act, out_pad=0, dila=1, groups=1, eps=1e-8, mom=0.9):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, output_padding=out_pad, dilation=dila,
                                       groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_ch, eps=eps, momentum=mom)
        self.act = act

    def forward(self, x):
        y = x
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        return y


class DwConvBnAct2D(nn.Module):
    def __init__(self, in_ch, depth_multiplier, ker_sz, stride, pad, act, dila=1, eps=1e-8, mom=0.9):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch*depth_multiplier, ker_sz, stride, pad, dilation=dila, groups=in_ch, bias=False)
        self.norm = nn.BatchNorm2d(in_ch*depth_multiplier, eps=eps, momentum=mom)
        self.act = act

    def forward(self, x):
        y = x
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        return y
