import numpy as np
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from collections import Iterable as _Iterable

from typing import List, Union


def get_padding_by_name(ker_sz, name='same'):
    if name.lower() == 'same':
        pad = np.int32(np.array(ker_sz) // 2).tolist()
    elif name.lower() == 'valid':
        pad = 0
    else:
        raise AssertionError(': "{}" is not expected'.format(name))
    return pad


def fixup_init(w, ker_sz, out_ch, fixup_l=12):
    with torch.no_grad():
        k = np.prod(ker_sz) * out_ch
        w.normal_(0, fixup_l ** (-0.5) * np.sqrt(2. / k))


def print_params_size(parameter, dtype_size=4):
    params_count = 0
    for p in parameter:
        params_count += np.prod(list(p.shape))
    print('params size %f MB' % (params_count * dtype_size / 1024 / 1024))
    return params_count


def print_params_size2(net):
    params = net.parameters()
    size = 0
    for p in params:
        # if p.dtype in (torch.int8, torch.uint8, torch.bool):
        #     ds = 1
        # elif p.dtype in (torch.float16, torch.int16): #, torch.uint16):
        #     ds = 2
        # elif p.dtype in (torch.float32, torch.int32): #, torch.uint32):
        #     ds = 4
        # elif p.dtype in (torch.float64, torch.int64): #, torch.uint64):
        #     ds = 8
        # else:
        #     raise RuntimeError('Unknow type', str(p.dtype))
        # size += p.numel() * ds
        size += p.numel() * p.element_size()
    print('params size %f MB' % (size / 1024 / 1024))
    return size


def _pair(ker_sz):
    if isinstance(ker_sz, int):
        return ker_sz, ker_sz
    elif isinstance(ker_sz, _Iterable) and len(ker_sz) == 2:
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
        elif issubclass(type(net_or_params), _Iterable):
            for p in net_or_params:
                p.clamp_(range[0], range[1])
        else:
            raise AssertionError('weight_clip: Unsupport type')
