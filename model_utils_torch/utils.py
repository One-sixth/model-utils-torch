import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from typing import List, Union, Iterable
from contextlib import contextmanager


def get_padding_by_name(ker_sz, name='same'):
    if name.lower() == 'same':
        pad = np.array(ker_sz, np.int) // 2
    elif name.lower() == 'valid':
        pad = 0
    else:
        raise AssertionError(': "{}" is not expected'.format(name))
    if isinstance(pad, Iterable):
        pad = tuple(pad)
    return pad


def fixup_init(w, ker_sz, out_ch, fixup_l=12):
    with torch.no_grad():
        k = np.prod(ker_sz) * out_ch
        w.normal_(0, fixup_l ** (-0.5) * np.sqrt(2. / k))


def print_params_size(module):
    params = module.parameters()
    size = 0
    count = 0
    for p in params:
        size += p.numel() * p.element_size()
        count += p.numel()
    print('params count {} size {} MB'.format(count, size / 1024 / 1024))
    return size



def print_buffers_size(module):
    buffers = module.buffers()
    size = 0
    count = 0
    for p in buffers:
        size += p.numel() * p.element_size()
        count += p.numel()
    print('buffers count {} size {} MB'.format(count, size / 1024 / 1024))
    return size


# from torch.nn.modules.utils import _pair
def _pair(ker_sz):
    if isinstance(ker_sz, int):
        return ker_sz, ker_sz
    elif isinstance(ker_sz, Iterable) and len(ker_sz) == 2:
        return tuple(ker_sz)
    else:
        raise AssertionError('Wrong kernel_size')


def weight_clip_(net_or_params: Union[nn.Module, List], range=(-1, 1), end_with_name: str='weight'):
    with torch.no_grad():
        if issubclass(type(net_or_params), (nn.Module, torch.jit.ScriptModule)):
            n: str
            p: nn.parameter.Parameter
            for n, p in net_or_params.named_parameters():
                if n.endswith(end_with_name):
                    p.clamp_(range[0], range[1])
        elif issubclass(type(net_or_params), Iterable):
            for p in net_or_params:
                p.clamp_(range[0], range[1])
        else:
            raise AssertionError('weight_clip: Unsupport type')


def module_weight_ema(ema_m: nn.Module, ref_m: nn.Module, decay=0.9999):
    '''
    对模型权重进行 指数移动平均值 操作

    来自：https://github.com/basiclab/GNGAN-PyTorch/blob/master/utils.py#L36

    :param ema_m: 要滑动更新的模型，一般指更新很慢的模型
    :param ref_m: 引用模型，一般指更新很快的那个模型
    :param decay: 滑动率，值越大，滑动得越慢，范围是 [0, 1]
    :return:
    '''
    assert 0 <= decay <= 1
    ref_dict = ref_m.state_dict()
    ema_dict = ema_m.state_dict()
    for key in ref_dict.keys():
        ema_dict[key].data.copy_(
            ema_dict[key].data * decay +
            ref_dict[key].data * (1 - decay)
        )


@contextmanager
def module_param_no_grad(m: nn.Module):
    '''
    用于阻止模块内的变量记录梯度，可以节省少量的显存和计算时间
    离开作用域后自动恢复

    使用方法:
    net = ...
    with module_no_grad(net):
        # 此时 net 内所有 params 的 requires_grad 将会设为False
        y = net(x)
    # 离开作用域，此时 net 内所有 params 的 requires_grad 将会恢复原来的设定

    来自：https://github.com/basiclab/GNGAN-PyTorch/blob/master/utils.py#L46

    :param m:
    :return:
    '''
    # 记录原始 requires_grad 设定
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    # 进入with作用域
    yield m
    # 离开with作用域
    # 恢复原始 requires_grad 设定
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])


def get_optim_cur_lr(optim: torch.optim.Optimizer, reduce='mean'):
    '''
    获得优化器当前的学习率
    :return: 是一个列表，代表每个参数组的学习率
    '''
    assert reduce in ('mean', 'sum', 'none'), f'Error! Bad reduce param {reduce}'
    lrs = []
    for group in optim.param_groups:
        lrs.append(group['lr'])

    if reduce == 'mean':
        lr = sum(lrs) / len(lrs)
    elif reduce == 'sum':
        lr = sum(lrs)
    else:
        lr = lrs

    return lr
