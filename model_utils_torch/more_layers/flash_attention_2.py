'''
FLASH-QUAD注意力层 2

注意：由于各种变体，改进一直在更新，这里仅提供参考实现。
为了方便使用新技术，本实现的API并不稳定，请复制本文件到你的项目里面，以确保稳定复现

注意2：FlashQuadCrossAttention 是我根据原FlashQuad设计结合标准的交叉注意力层推出来的，原型没有该设计，不能确定有效性。
注意3：FlashQuadSelfAttention 与原实现相比，去除了基于序列长度缩放，并且 q_gamma, k_gamma 的初始化方法也做了修改。

增加自注意力层和交叉自注意力层
https://spaces.ac.cn/archives/8934

参考
https://github.com/lucidrains/FLASH-pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Union, Callable
from ..ops import make_sinusoidal_position_embedding, apply_rotary_position_embedding


__all__ = ['FlashQuadCrossAttention', 'FlashQuadSelfAttention']


_TensorOptional = Union[None, torch.Tensor]


@torch.jit.script
def laplacian_attn_fn(x):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """
    mu = math.sqrt(0.5)
    std = math.sqrt(0.25 * math.pi)
    return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5


@torch.jit.script
def flash_quad_cross_attention(q, k, v, attn_mul: _TensorOptional=None, attn_bias: _TensorOptional=None, attn_act_fn: str='laplacian'):
    assert attn_act_fn in ('laplacian', 'relu2'), 'Error! Invalid param attn_act_fn: ' + attn_act_fn
    a = q @ k.transpose(-1, -2)
    # qk shape [B, Q, K]

    # scale
    a /= math.sqrt(q.shape[1] * k.shape[1])

    if attn_mul is not None:
        a *= attn_mul

    if attn_bias is not None:
        a += attn_bias

    if attn_act_fn == 'laplacian':
        a = laplacian_attn_fn(a)
    elif attn_act_fn == 'relu2':
        a = F.relu(a) ** 2

    o = a @ v
    return o


def _get_norm(use_norm, in_dim):
    if use_norm == 'layernorm' or use_norm is True:
        norm = nn.LayerNorm(in_dim)
    elif use_norm == 'layernorm_nobias':
        norm = nn.LayerNorm(in_dim)
        norm.register_parameter('bias', None)
    elif use_norm in [None, False, 'none', 'identity']:
        norm = nn.Identity()
    elif isinstance(use_norm, nn.Module):
        norm = use_norm
    elif isinstance(use_norm, Callable):
        norm = use_norm(in_dim)
    else:
        raise AssertionError('Error! Invalid norm type.')
    return norm


class FlashQuadCrossAttention(torch.jit.ScriptModule):
    __constants__ = ['use_skip', 'attn_act_fn']

    def __init__(self, in_dim, out_dim, expand_dim, squeeze_dim, use_norm='layernorm', use_skip=True, attn_act_fn='laplacian', use_rotary_pos_emb=False, linear_layer_nobias=False):
        super().__init__()
        warnings.warn(f'Warning! You are using the {self.__class__.__name__} temporary implementation.')

        self.use_skip = use_skip
        self.attn_act_fn = attn_act_fn
        if use_skip:
            assert in_dim == out_dim
        self.use_rotary_pos_emb = use_rotary_pos_emb

        self.norm = _get_norm(use_norm, in_dim)
        self.norm2 = _get_norm(use_norm, in_dim)

        self.q_m = nn.Linear(in_dim, squeeze_dim, bias=not linear_layer_nobias)
        self.k_m = nn.Linear(in_dim, squeeze_dim, bias=not linear_layer_nobias)
        self.v_m = nn.Linear(in_dim, expand_dim, bias=not linear_layer_nobias)
        self.u_m = nn.Linear(in_dim, expand_dim, bias=not linear_layer_nobias)

        self.q_gamma = nn.Parameter(torch.rand(squeeze_dim) / math.sqrt(squeeze_dim))
        self.q_bias = nn.Parameter(torch.zeros(squeeze_dim))
        self.k_gamma = nn.Parameter(torch.rand(squeeze_dim) / math.sqrt(squeeze_dim))
        self.k_bias = nn.Parameter(torch.zeros(squeeze_dim))

        self.out = nn.Linear(expand_dim, out_dim, bias=not linear_layer_nobias)

    @torch.jit.script_method
    def forward(self, _q, _k, attn_mul: _TensorOptional=None, attn_bias: _TensorOptional=None, rotary_pos_start: int=0, rotary_pos_scale: int=1):
        z1 = self.norm(_q)
        z2 = self.norm2(_k)

        q = self.q_m(z1)
        k = self.k_m(z2)
        v = self.v_m(z2)
        u = self.u_m(z1)

        q = q * self.q_gamma + self.q_bias
        k = k * self.k_gamma + self.k_bias

        if self.use_rotary_pos_emb:
            qL, kL = q.shape[1], k.shape[1]
            pos_emb = make_sinusoidal_position_embedding(max(qL, kL), pos_ch=max(q.shape[2], k.shape[2]), pos_start=rotary_pos_start, pos_scale=rotary_pos_scale, device=q.device)
            q = apply_rotary_position_embedding(q, pos_emb[:qL, :q.shape[2]])
            k = apply_rotary_position_embedding(k, pos_emb[:kL, :k.shape[2]])

        y = u * flash_quad_cross_attention(q, k, v, attn_mul, attn_bias, self.attn_act_fn)

        y = self.out(y)
        if self.use_skip:
            y = _q + y

        return y


class FlashQuadSelfAttention(torch.jit.ScriptModule):
    __constants__ = ['use_skip', 'attn_act_fn']

    def __init__(self, in_dim, out_dim, expand_dim, squeeze_dim, use_norm='layernorm', use_skip=True, attn_act_fn='laplacian', use_rotary_pos_emb=False, linear_layer_nobias=False):
        super().__init__()
        warnings.warn(f'Warning! You are using the {self.__class__.__name__} temporary implementation.')

        self.use_skip = use_skip
        self.attn_act_fn = attn_act_fn
        if use_skip:
            assert in_dim == out_dim
        self.use_rotary_pos_emb = use_rotary_pos_emb

        self.norm = _get_norm(use_norm, in_dim)

        self.u_m = nn.Linear(in_dim, expand_dim, bias=not linear_layer_nobias)
        self.v_m = nn.Linear(in_dim, expand_dim, bias=not linear_layer_nobias)
        self.z_m = nn.Linear(in_dim, squeeze_dim, bias=not linear_layer_nobias)

        self.q_gamma = nn.Parameter(torch.rand(squeeze_dim) / math.sqrt(squeeze_dim))
        self.q_bias = nn.Parameter(torch.zeros(squeeze_dim))
        self.k_gamma = nn.Parameter(torch.rand(squeeze_dim) / math.sqrt(squeeze_dim))
        self.k_bias = nn.Parameter(torch.zeros(squeeze_dim))

        self.out = nn.Linear(expand_dim, out_dim, bias=not linear_layer_nobias)

    @torch.jit.script_method
    def forward(self, x, attn_mul: _TensorOptional=None, attn_bias: _TensorOptional=None, rotary_pos_start: int=0, rotary_pos_scale: int=1):
        y = self.norm(x)

        u = self.u_m(y)
        v = self.v_m(y)
        z = self.z_m(y)

        q = z * self.q_gamma + self.q_bias
        k = z * self.k_gamma + self.k_bias

        if self.use_rotary_pos_emb:
            qL, kL = q.shape[1], k.shape[1]
            pos_emb = make_sinusoidal_position_embedding(max(qL, kL), pos_ch=max(q.shape[2], k.shape[2]), pos_start=rotary_pos_start, pos_scale=rotary_pos_scale, device=q.device)
            q = apply_rotary_position_embedding(q, pos_emb[:qL, :q.shape[2]])
            k = apply_rotary_position_embedding(k, pos_emb[:kL, :k.shape[2]])

        y = u * flash_quad_cross_attention(q, k, v, attn_mul, attn_bias, self.attn_act_fn)

        y = self.out(y)
        if self.use_skip:
            y = x + y

        return y
