'''
FLASH-QUAD注意力层
https://spaces.ac.cn/archives/8934
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union


__all__ = ['FlashQuadAttention']


# @torch.jit.script
# def make_mask(mask: torch.Tensor, L: int):
#     '''
#     更加不同的输入情况，制作掩码
#     支持 1D，2D，3D输入
#     1D 输入时要求为long类型
#     2D 和 3D 输入时，要求为bool类型
#     [B, L1, L2]
#     B 代表每句话的维度
#     L1 代表每个字的维度
#     L2 代表每个字对其他所有的字是否可见，1或True代表可见，0或False代表不可见
#     :param mask:
#     :param L:
#     :return:
#     '''
#     if mask.ndim == 1:
#         # mask [B]
#         # 这个代表每批数据（每句话）用从头到末尾不同长度的掩码，每个字用相同的掩码
#         assert mask.dtype == torch.long, 'Error! mask has wrong dtype.'
#         m = torch.arange(L)[None,].repeat(mask.shape[0], 1)
#         # m [B, L]
#         m = m < mask[:, None]
#         # m [B, 1, L]
#         m = m[:, None, :]
#     elif mask.ndim == 2:
#         # mask [B, L]
#         # 这个代表每批数据（每句话）用不同的掩码，但每个字都用同样的掩码
#         assert mask.dtype == torch.bool, 'Error! mask has wrong dtype.'
#         assert mask.shape[1] == L, 'Error! mask has bad shape.'
#         # m [B, 1, L]
#         m = mask[:, None, :]
#     elif mask.ndim == 3:
#         # mask [B, L, L]
#         # 这个代表每批数据（每句话）的每个字都用不同的掩码
#         assert mask.dtype == torch.bool, 'Error! mask has wrong dtype.'
#         assert mask.shape[1] == mask.shape[2] == L, 'Error! mask has bad shape.'
#         # m [B, L, L]
#         m = mask
#     else:
#         raise AssertionError('Error! mask has bad shape.')
#
#     m = torch.broadcast_to(m, [mask.shape[0], L, L])
#     return m


class attn(torch.jit.ScriptModule):
    def __init__(self, in_dim, s_dim):
        super().__init__()
        self.z_m = nn.Linear(in_dim, s_dim)
        self.q_gamma = nn.Parameter(torch.rand(s_dim) / math.sqrt(s_dim))
        self.q_bias = nn.Parameter(torch.zeros(s_dim))
        self.k_gamma = nn.Parameter(torch.rand(s_dim) / math.sqrt(s_dim))
        self.k_bias = nn.Parameter(torch.zeros(s_dim))

    @torch.jit.script_method
    def forward(self, x, v):
        z = self.z_m(x)
        q = z * self.q_gamma + self.q_bias
        k = z * self.k_gamma + self.k_bias
        qk = q @ k.transpose(-1, -2)
        a = F.relu(qk) ** 2
        o = a @ v
        return o


class FlashQuadAttention(torch.jit.ScriptModule):
    __constants__ = ['use_skip']

    def __init__(self, in_dim, out_dim, expand_dim, squeeze_dim, use_norm=True, use_skip=True):
        super().__init__()
        self.use_skip = use_skip
        if use_norm:
            self.norm = nn.LayerNorm(in_dim)
        else:
            self.norm = nn.Identity()
        self.u_m = nn.Linear(in_dim, expand_dim)
        self.v_m = nn.Linear(in_dim, expand_dim)
        self.attn = attn(in_dim, squeeze_dim)
        if out_dim is None:
            out_dim = in_dim
        self.out = nn.Linear(expand_dim, out_dim)

    @torch.jit.script_method
    def forward(self, x):
        y = self.norm(x)

        u = self.u_m(y)
        v = self.v_m(y)

        y = u * self.attn(y, v)

        y = self.out(y)
        if self.use_skip:
            y = x + y
        return y


def test_flash_attention():
    x = torch.rand(3, 200, 128)
    # x [B, L, C]
    m = FlashQuadAttention(128, 128, 256, 128)
    y = m(x)
    # print(m.code)
    # print(make_mask.code)
    print(y.shape)

    r1 = y.shape == (3, 200, 128)

    # mask1d = torch.randint(0, 200, [3,])
    # mask2d = torch.randint(0, 200, [3, 200]) < 100
    # mask3d = torch.randint(0, 200, [3, 200, 200]) < 100
    #
    # y1d = m(x, mask1d)
    # y2d = m(x, mask2d)
    # y3d = m(x, mask3d)
    # print(y1d.shape)
    # print(y2d.shape)
    # print(y3d.shape)
    #
    # r2 = y1d.shape == y2d.shape == y3d.shape == (3, 200, 128)

    r = r1 #and r2
    return r


if __name__ == '__main__':
    test_flash_attention()