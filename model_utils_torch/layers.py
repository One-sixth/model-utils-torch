'''
预定义组合层，目的为便于使用

尽可能使用jit编译，如果jit有困难，则果断不使用jit

'''

import torch
import torch.jit

import torch.nn as nn
import torch.nn.functional as F
import math

from . import ops
from . import utils
from .more_layers import *


'''
注意写 torch.jit.script 时需要手动添加非 Tensor 参数的注释
'''


class Interpolate(torch.jit.ScriptModule):
    '''
    与Upsample层等价
    '''
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name', 'recompute_scale_factor']

    def __init__(self, size=None, scale_factor=None, mode: str='nearest', align_corners=None, recompute_scale_factor=None, antialias=False) -> None:
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    @torch.jit.script_method
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=self.recompute_scale_factor, antialias=self.antialias)

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


Upsample = Interpolate


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


# class RmsNorm(torch.jit.ScriptModule):
#     def __init__(self, in_feat, dim, eps=1e-8):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(in_feat))
#         self.dim = dim
#         self.eps = eps
#
#     @torch.jit.script_method
#     def forward(self, x: torch.Tensor):
#         y = x / ops.root_mean_square(x, self.dim, True, self.eps) * self.weight
#         return y


class ScaleNorm(torch.jit.ScriptModule):
    def __init__(self, scale=1., dim=-1, eps=1e-8):
        super().__init__()
        if isinstance(dim, int):
            dim = (dim,)
        assert dim is None or all([isinstance(i, int) for i in dim]), 'Error! 参数 dim 只允许 None，整数，整数列表'
        self.dim = dim
        self.weight = nn.Parameter(torch.as_tensor(scale))
        self.eps = eps

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        y = x / ops.root_mean_square(x, self.dim, True, self.eps) * self.weight
        return y


class GradScale(nn.Module):
    def __init__(self, scale=1.):
        '''
        梯度缩放，方法1，不能jit，可以等价替换为GradScale2从而允许jit
        :param scale:
        '''
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            return ops.grad_scale(x, self.scale)
        else:
            return x


class GradScale2(torch.jit.ScriptModule):
    def __init__(self, scale=1.):
        '''
        梯度缩放，方法2，可以直接jit
        :param scale:
        '''
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            return ops.grad_scale_2(x, self.scale)
        else:
            return x


class SwiGLU(torch.jit.ScriptModule):
    def __init__(self, in_feat, ffn_mul, out_feat=None, no_bias=False):
        super().__init__()
        assert in_feat * ffn_mul % 2 == 0
        if out_feat is None:
            out_feat = in_feat
        hidden_dim = int(round(in_feat * ffn_mul))

        # 注意，bias 可以有效帮助拟合，建议保留
        self.layer1 = nn.Linear(in_feat, hidden_dim * 2, bias=not no_bias)
        self.layer2 = nn.Linear(hidden_dim, out_feat, bias=not no_bias)
        self.reset_parameters()

    def reset_parameters(self):
        limit = 1 / math.sqrt(self.layer1.weight.shape[0])
        nn.init.uniform_(self.layer1.weight, -limit, limit)
        self.layer2.weight.data.zero_()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        # x shape [B, L, C]
        gate, feat = self.layer1(x).chunk(2, -1)
        feat = F.silu(gate) * feat
        out = self.layer2(feat)
        return out
