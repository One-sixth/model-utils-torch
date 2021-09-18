import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from typing import List, Union, Iterable


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
