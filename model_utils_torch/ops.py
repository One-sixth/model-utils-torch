import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

# from torch.nn import conv, Linear
# from torch.nn.utils import _pair

# from .utils import *

from typing import Tuple
try:
    from .more_ops import *
except (ModuleNotFoundError, ImportError):
    from more_ops import *


@torch.jit.script
def channel_shuffle(x, n_group: int):
    """
    通道扰乱操作
    """
    B, C = x.shape[:2]
    shape2 = list(x.shape[2:])
    x = x.reshape([B, n_group, C // n_group] + shape2)
    x = x.transpose(1, 2)
    x = x.reshape([B, C] + shape2)
    return x


@torch.jit.script
def resize_ref(x, shortpoint, method: str = 'bilinear', align_corners: bool = None):
    """
    :type x: torch.Tensor
    :type shortpoint: torch.Tensor
    :type method: str
    :type align_corners: bool
    """
    hw = shortpoint.shape[2:4]
    ihw = x.shape[2:4]
    if hw != ihw:
        x = torch.nn.functional.interpolate(x, hw, mode=method, align_corners=align_corners)
    return x


@torch.jit.script
def add_coord(x: torch.Tensor):
    """
    增加两层坐标层
    """
    b, c, h, w = x.shape

    y_coord = torch.linspace(-1, 1, h, dtype=x.dtype, device=x.device)
    y_coord = y_coord.reshape(1, 1, -1, 1)
    y_coord = y_coord.repeat(b, 1, 1, w)

    x_coord = torch.linspace(-1, 1, w, dtype=x.dtype, device=x.device)
    x_coord = x_coord.reshape(1, 1, 1, -1)
    x_coord = x_coord.repeat(b, 1, h, 1)

    o = torch.cat((x, y_coord, x_coord), 1)
    return o


@torch.jit.script
def pixelwise_norm(x, eps: float = 1e-8):
    """
    Pixelwise feature vector normalization.
    :param x: input activations volume
    :param eps: small number for numerical stability
    :return: y => pixel normalized activations
    """
    return x * x.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()


@torch.jit.script
def flatten(x):
    y = x.reshape(x.shape[0], -1)
    return y


@torch.jit.script
def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])

    ss = style_feat.shape
    cs = content_feat.shape

    style_mean = style_feat.mean((2, 3), keepdim=True)
    style_std = style_feat.reshape(ss[0], ss[1], -1).std(2, unbiased=False).reshape_as(style_mean)
    content_mean = content_feat.mean((2, 3), keepdim=True)
    content_std = content_feat.reshape(cs[0], cs[1], -1).std(2, unbiased=False).reshape_as(content_mean)

    normalized_feat = (content_feat - content_mean) / (content_std + 1e-8)
    return normalized_feat * style_std + style_mean


# mod from https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
@torch.jit.script
def minibatch_stddev(x, group_size: int = 4, num_new_features: int = 1, eps: float = 1e-8):
    group_size = group_size if group_size < x.shape[0] else x.shape[0]
    s = x.shape
    y = x.reshape(group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3])
    y = y - y.mean(dim=0, keepdim=True)
    y = y.pow(2).mean(dim=0)
    y = (y + eps).sqrt()
    y = y.mean(dim=(2, 3, 4), keepdim=True)
    y = y.mean(dim=2)
    y = y.repeat(group_size, 1, s[2], s[3])
    return torch.cat((x, y), dim=1)


@torch.jit.script
def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC // (pH * pW), iH * pH, iW * pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)  # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


@torch.jit.script
def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC * (pH * pW), iH // pH, iW // pW
    y = y.reshape(B, iC, oH, pH, oW, pW)
    y = y.permute(0, 1, 3, 5, 2, 4)  # B, iC, pH, pW, oH, oW
    y = y.reshape(B, oC, oH, oW)
    return y


@torch.jit.script
def one_hot(class_array: torch.Tensor, class_num: int, dim: int = -1, dtype: torch.dtype = torch.int32):
    '''
    可将[D1, D2, D3, ..., DN] 矩阵转换为 [D1, D2, ..., DN, D(N+1)] 的独热矩阵
    :param class_array: [D1, D2, ..., DN] 类别矩阵
    :param class_num:   类别数量
    :param dim:
    :param dtype:
    :return: y => onehot array
    '''
    a = torch.arange(class_num, dtype=torch.int32, device=class_array.device)
    for _ in range(class_array.ndim):
        a = torch.unsqueeze(a, 0)
    b = (class_array[..., None] == a).to(dtype)
    if dim != -1:
        b = torch.movedim(b, -1, dim)
    return b


@torch.jit.script
def one_hot_invert(onehot_array, dim: int = -1, dtype: torch.dtype = torch.int32):
    '''
    上面one_hot的逆操作
    可将[D1, D2, D3, ..., DN] 的独热矩阵转换为 [D1, D2, ..., D(N-1)] 的类别矩阵
    :param onehot_array: [D1, D2, ..., DN] 独热矩阵
    :param dim:
    :param dtype:
    :return: y => class array
    '''
    class_arr = torch.max(onehot_array, dim)[1]
    class_arr = class_arr.to(dtype)
    return class_arr


@torch.jit.script
def linspace_grid(sizes: list[int], value_ranges: list[float] = (-1., 1.), stack_dim: int = 0):
    '''
    可以快速生成一个数字坐标网格
    :param sizes: 每层的大小
    :param value_ranges: 支持输入方法 将值域应用到全部层[-1,1] 和 每层均指定特定的值域[-1,1,-1,1,...]
    :param stack_dim:
    :return:
    '''

    # 检查参数sizes
    sizes = [int(round(s)) for s in sizes]
    value_ranges = [float(v) for v in value_ranges]

    # 检查参数 value_ranges
    if len(value_ranges) == 2:
        value_ranges = value_ranges * len(sizes)

    assert len(value_ranges) == 2 * len(sizes)

    # 开始生成网格
    ds = []
    for i in range(len(sizes)):
        r1 = i * 2
        r2 = r1 + 1
        d = torch.linspace(value_ranges[r1], value_ranges[r2], sizes[i])
        ds.append(d)

    o = torch.meshgrid(ds, indexing='ij')
    o = torch.stack(o, stack_dim)

    return o


class _GradScaleOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        s, = ctx.saved_tensors
        return grad_output * s, None


grad_scale = _GradScaleOp.apply


@torch.jit.script
def make_sinusoidal_position_channel_embedding(pos_ch: int=2, cycle: float=10000., device: torch.device= 'cpu'):
    '''
    生成通道嵌入，用于给 make_sinusoidal_position_embedding 加速
    :param pos_ch:  通道数
    :param cycle:   周期
    :param device:  设备
    :return:
    '''
    ch_idx = torch.arange(pos_ch, device=device)
    ch_emb = torch.pow(cycle, 2 * (ch_idx // 2) / pos_ch)
    return ch_emb


@torch.jit.script
def make_sinusoidal_position_embedding(len: int, pos_start: int=0, pos_ch: int=2, cycle: float=10000., pos_scale: int=1, device: torch.device= 'cpu',
                                       ch_emb: Union[None, torch.Tensor]=None):
    '''
    sinusoidal_position_embedding 正余弦绝对位置编码
    :param len:         长度
    :param pos_start:   起始位置
    :param pos_ch:      通道数
    :param cycle:       周期
    :param pos_scale:   坐标缩放
    :param device:      设备
    :param ch_emb:      通道嵌入，由 make_sinusoidal_position_channel_embedding 生成，用于加速。
    :return:            输出 Tensor shape [len, pos_ch]
    '''
    if ch_emb is None:
        ch_emb = make_sinusoidal_position_channel_embedding(pos_ch, cycle, device)
    else:
        assert ch_emb.ndim == 1 and ch_emb.shape[0] == pos_ch, 'Error! Bad ch_emb shape.'

    # pos = torch.arange(len, device=device).mul(pos_scale).add(pos_start)[:, None]
    pos = torch.arange(pos_start, pos_scale*len+pos_start, pos_scale, device=device)[:, None]
    pos_emb = pos / ch_emb[None,]

    torch.sin_(pos_emb[:, 0::2])
    torch.cos_(pos_emb[:, 1::2])

    # shape [len, pos_ch]
    return pos_emb


@torch.jit.script
def apply_rotary_position_embedding(x, sin_pos_emb):
    '''
    旋转绝对位置编码 rotary_position_embedding
    该编码是乘性编码，且需要通道间交互
    要求输入通道数大于2并且为2的倍数
    :param x:           shape [..., L, C] 输入Tensor
    :param sin_pos_emb: shape [L, C] 正余弦绝对位置编码，由 make_sinusoidal_position_embedding 生成
    :return:
    '''
    assert x.shape[-1] % 2 == 0 and x.shape[-1] >= 2, 'Error! The input channels is required to be greater than 2 and a multiple of 2.'
    sin, cos = sin_pos_emb[..., 0::2], sin_pos_emb[..., 1::2]
    x1, x2 = x[..., 0::2], x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    # o1,o2 shape [..., L, C//2]

    o = torch.stack([o1, o2], -1)
    # o shape [..., L, C//2, 2]
    o = o.flatten(-2)
    # o shape [..., L, C]
    return o
