'''
改自 https://github.com/switchablenorms/Switchable-Normalization/blob/master/devkit/ops/switchable_norm.py
SwitchableNorm 和 SwitchableNorm1D 可能会有点问题，这里的IN就是他们自身。

2021-9-12
修改了几句，使其兼容 torch.jit.script
'''

import torch
import torch.nn as nn


class SwitchableNormND(nn.Module):
    def __init__(self, N, num_features, eps=1e-8, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__()
        assert N >= 0
        self.N = N
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1), True)
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1), True)

        weight_num = 2 + (1 if using_bn else 0)
        self.mean_weight = nn.Parameter(torch.ones(weight_num), True)
        self.var_weight = nn.Parameter(torch.ones(weight_num), True)

        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.ndim-2 != self.N:
            raise ValueError('expected {}D input (got {}D input)'.format(self.N, input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        B, C = x.shape[:2]
        shape2 = list(x.shape[2:])

        x = x.reshape(B, C, -1)

        mean_in = x.mean(2, keepdim=True)
        var_in = x.var(2, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
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

        x = (x - mean) / (var + self.eps).sqrt()
        x = x * self.weight + self.bias
        x = x.reshape([B, C] + shape2)
        return x


class SwitchableNorm(SwitchableNormND):
    def __init__(self, num_features, eps=1e-8, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__(0, num_features, eps, momentum, using_moving_average, using_bn, last_gamma)


class SwitchableNorm1D(SwitchableNormND):
    def __init__(self, num_features, eps=1e-8, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__(1, num_features, eps, momentum, using_moving_average, using_bn, last_gamma)


class SwitchableNorm2D(SwitchableNormND):
    def __init__(self, num_features, eps=1e-8, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__(2, num_features, eps, momentum, using_moving_average, using_bn, last_gamma)


class SwitchableNorm3D(SwitchableNormND):
    def __init__(self, num_features, eps=1e-8, momentum=0.9, using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__(3, num_features, eps, momentum, using_moving_average, using_bn, last_gamma)
