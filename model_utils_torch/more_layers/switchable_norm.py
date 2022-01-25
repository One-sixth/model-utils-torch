'''
改自 https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py
SwitchableNorm 和 SwitchableNorm1D 可能会有点问题，这里的IN就是他们自身。

2021-12-21
变更 momentum 参数定义，现在定义与nn.BatchNorm一致，值越小，更新就越慢
修改参数 zero_gamma 和 affine 到 gamma_init 和 bias_init，自定义能力加强

2021-9-12
修改了几句，使其兼容 torch.jit.script
'''

import torch
import torch.nn as nn
import math


__all__ = ['SwitchableNorm', 'SwitchableNorm1D', 'SwitchableNorm2D', 'SwitchableNorm3D', 'SwitchableNormND']


class SwitchableNormND(nn.Module):
    def __init__(self, N, num_features, eps=1e-8, momentum=0.1, using_moving_average=True, using_bn=True, gamma_init=1., bias_init=0.):
        super().__init__()
        assert N >= 0
        self.N = N
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.gamma_init = gamma_init
        self.bias_init = bias_init

        if gamma_init is not None:
            self.weight = nn.Parameter(torch.full([1, num_features, 1], gamma_init), True)
        else:
            self.register_buffer('weight', None)

        if bias_init is not None:
            self.bias = nn.Parameter(torch.full([1, num_features, 1], bias_init), True)
        else:
            self.register_buffer('bias', None)

        weight_num = 2 + (1 if using_bn else 0)
        self.mean_weight = nn.Parameter(torch.ones(weight_num), True)
        self.var_weight = nn.Parameter(torch.ones(weight_num), True)

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def _check_input_dim(self, input):
        if input.ndim-2 != self.N:
            raise ValueError('expected {}D input (got {}D input)'.format(self.N, input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        B, C = x.shape[:2]
        shape2 = list(x.shape[2:])
        # N 个像素
        if x.ndim == 2:
            N = 1
        else:
            N = math.prod(shape2)

        x = x.reshape(B, C, N)

        if N >= 2:
            mean_in = x.mean(2, keepdim=True)
            var_in = x.var(2, keepdim=True)
        else:
            # 如果 N 小于2，则IN无意义
            mean_in = torch.zeros(1, device=x.device)
            var_in = torch.ones(1, device=x.device)

        if C*N >= 2:
            mean_ln = x.mean([1, 2], keepdim=True)
            var_ln = x.var([1, 2], keepdim=True)
        else:
            # 如果 N*C 小于2，则LN无意义
            mean_ln = torch.zeros(1, device=x.device)
            var_ln = torch.ones(1, device=x.device)

        if self.using_bn:
            if self.training:
                # 如果 B*C 小于2，则BN无意义
                if B*C >= 2:
                    mean_bn = x.mean([0, 2], keepdim=True)
                    var_bn = x.var([0, 2], keepdim=True)
                else:
                    mean_bn = torch.zeros(1, device=x.device)
                    var_bn = torch.ones(1, device=x.device)

                if self.using_moving_average:
                    self.running_mean.mul_(1 - self.momentum)
                    self.running_mean.add_(self.momentum * mean_bn.data)
                    self.running_var.mul_(1 - self.momentum)
                    self.running_var.add_(self.momentum * var_bn.data)
                else:
                    self.running_mean.set_(mean_bn.data)
                    self.running_var.set_(var_bn.data)
            else:
                mean_bn = self.running_mean
                var_bn = self.running_var
        else:
            # 本段用于兼容 torch.jit.script，实际无任何作用
            mean_bn = torch.zeros(1, device=x.device)
            var_bn = torch.zeros(1, device=x.device)

        mean_weight = torch.softmax(self.mean_weight, 0)
        var_weight = torch.softmax(self.var_weight, 0)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x - mean) * (var + self.eps).rsqrt()

        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias

        x = x.reshape([B, C] + shape2)
        return x


class SwitchableNorm(SwitchableNormND):
    def __init__(self, *args, **kwargs):
        super().__init__(0, *args, **kwargs)


class SwitchableNorm1D(SwitchableNormND):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)


class SwitchableNorm2D(SwitchableNormND):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)


class SwitchableNorm3D(SwitchableNormND):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
