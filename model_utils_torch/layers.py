'''
预定义组合层，目的为便于使用

尽可能使用jit编译，如果jit有困难，则果断不使用jit

'''

import torch
import torch.jit

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Iterable as _Iterable
from typing import Callable as _Callable

try:
    from . import ops
    from . import utils
    from .more_layers import *
except (ModuleNotFoundError, ImportError):
    import ops
    import utils
    from more_layers import *

'''
注意写 torch.jit.script 时需要手动添加非 Tensor 参数的注释
'''


class Upsample(torch.jit.ScriptModule):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='bilinear', align_corners=None):
        super().__init__()

        # scale_factor 不允许是整数，有点坑。。
        if size is None:
            if isinstance(scale_factor, _Iterable):
                scale_factor = tuple([float(i) for i in scale_factor])
            else:
                scale_factor = float(scale_factor)

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return F.interpolate(x, self.size, self.scale_factor, self.mode, self.align_corners)


class UpsampleConcat(torch.jit.ScriptModule):
    __constants__ = ['method', 'align_corners']

    def __init__(self, method='bilinear', align_corners=None):
        super().__init__()
        self.method = method
        self.align_corners = align_corners

    @torch.jit.script_method
    def forward(self, x, shortpoint):
        shape = shortpoint.shape
        x = F.interpolate(x, (shape[2], shape[3]), mode=self.method, align_corners=self.align_corners)
        x = torch.cat((x, shortpoint), 1)
        return x


class LinearGroup(torch.jit.ScriptModule):
    __constants__ = ['groups', 'use_bias']

    def __init__(self, in_feat, out_feat, groups, bias=True):
        super().__init__()
        self.groups = groups
        self.use_bias = bias
        in_feat_g = in_feat // groups
        out_feat_g = out_feat // groups

        assert in_feat_g * groups == in_feat, 'Found in_feat_g * groups != in_feat'
        assert out_feat_g * groups == out_feat, 'Found out_feat_g * groups != out_feat'

        self.weight = nn.Parameter(torch.zeros(groups, out_feat_g, in_feat_g), True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_feat), True)
        else:
            self.register_buffer('bias', torch.zeros(0))

    @torch.jit.script_method
    def forward(self, x):
        ys = torch.chunk(x, self.groups, -1)
        out_ys = []
        for i in range(self.groups):
            out_ys.append(F.linear(ys[i], self.weight[i]))
        y = torch.cat(out_ys, -1)
        if self.use_bias:
            y = y + self.bias
        return y


class AdaptiveGemPool(torch.jit.ScriptModule):
    __constants__ = ['dim', 'eps', 'keepdim']

    def __init__(self, dim=(2, 3), p=3, eps=1e-6, keepdim=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.keepdim = keepdim
        self.p = nn.Parameter(torch.ones(1) * p)

    @torch.jit.script_method
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(self.dim, keepdim=self.keepdim).pow(1. / self.p)


class Reshape(torch.jit.ScriptModule):
    __constants__ = ['shape']

    def __init__(self, new_shape):
        super().__init__()
        self.shape = tuple(int(i) for i in new_shape)

    @torch.jit.script_method
    def forward(self, x):
        return x.reshape(self.shape)

    def extra_repr(self):
        return "{shape}".format(**self.__dict__)
    
    
class InstanceReshape(torch.jit.ScriptModule):
    __constants__ = ['shape']

    def __init__(self, new_shape):
        super().__init__()
        self.shape = tuple(int(i) for i in new_shape)

    @torch.jit.script_method
    def forward(self, x):
        new_shape = (x.shape[0],) + self.shape
        return x.reshape(new_shape)

    def extra_repr(self):
        return "{shape}".format(**self.__dict__)

# class OctConv2D(_base_conv_setting):
#     def __init__(self, in_ch, out_ch, ker_sz=3, stride=1, pad='same', act=None, bias=True, groups=1, dila=1, alpha=(0.5, 0.5), *, use_fixup_init=False, norm_kwargs={}):
#         super().__init__(in_ch, out_ch, ker_sz, stride, pad, act, bias, dila)
#
#         self.act = act
#         if act is None:
#             self.act = Identity()
#
#         # 限定输入和输出必定存在高频特征
#         assert 0. <= np.min(alpha) and np.max(alpha) < 1, "Alphas should be in the interval from 0. to 1."
#         assert stride == 1 or stride == 2, "now only support stridt equal 1 or 2"
#         self.downscale = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         self.upscale = Upsample(scale_factor=2, mode='nearest')
#
#         alpha_in, alpha_out = alpha[0], alpha[1]
#         self.alpha_in, self.alpha_out = alpha_in, alpha_out
#
#         in_l_ch = int(alpha_in * in_ch)
#         in_h_ch = in_ch - in_l_ch
#         out_l_ch = int(alpha_out * out_ch)
#         out_h_ch = out_ch - out_l_ch
#
#         self.conv_l2l = nn.Conv2d(in_l_ch, out_l_ch, ker_sz, 1, self.pad, dila, groups, bias) \
#                         if alpha_in > 0 and alpha_out > 0 else None
#
#         self.conv_l2h = nn.Conv2d(in_l_ch, out_h_ch, ker_sz, 1, self.pad, dila, groups, bias) \
#                         if alpha_in > 0 and alpha_out < 1 else None
#
#         self.conv_h2l = nn.Conv2d(in_h_ch, out_l_ch, ker_sz, 1, self.pad, dila, groups, bias) \
#                         if alpha_in < 1 and alpha_out > 0 else None
#
#         self.conv_h2h = nn.Conv2d(in_h_ch, out_h_ch, ker_sz, 1, self.pad, dila, groups, bias) \
#                         if alpha_in < 1 and alpha_out < 1 else None
#
#         self.h_norm2d = Identity()
#         self.l_norm2d = Identity()
#
#         if isinstance(bias, _Callable):
#             self.h_norm2d = nn.Sequential(bias(out_h_ch, **norm_kwargs), act)
#             if alpha_out > 0:
#                 self.l_norm2d = nn.Sequential(bias(out_l_ch, **norm_kwargs), act)
#
#     def oct_forward(self, x_h, x_l):
#
#         if self.alpha_in == 0:
#             if self.stride > 1:
#                 x_h = self.downscale(x_h)
#             y_h2h = self.conv_h2h(x_h)
#             if self.alpha_out > 0:
#                 y_h2l = self.conv_h2l(self.downscale(x_h))
#                 return y_h2h, y_h2l
#             else:
#                 return y_h2h, None
#
#         elif self.alpha_in > 0:
#             if self.stride > 1:
#                 x_h = self.downscale(x_h)
#             y_h2h = self.conv_h2h(x_h)
#             y_l2h = self.conv_l2h(x_l)
#             if self.stride == 1:
#                 y_l2h = self.upscale(y_l2h)
#             y_h = y_h2h + y_l2h
#             if self.alpha_out > 0:
#                 y_h2l = self.conv_h2l(self.downscale(x_h))
#                 if self.stride > 1:
#                     x_l = self.downscale(x_l)
#                 y_l2l = self.conv_l2l(x_l)
#                 y_l = y_h2l + y_l2l
#                 return y_h, y_l
#             else:
#                 return y_h, None
#
#     # @torch.jit.script_method
#     def forward(self, x):
#         x_h, x_l = x if isinstance(x, (tuple, list)) else (x, None)
#
#         y_h, y_l = self.oct_forward(x_h, x_l)
#         y_h = self.h_norm2d(y_h)
#         if y_l is not None:
#             y_l = self.l_norm2d(y_l)
#             return y_h, y_l
#         return y_h
