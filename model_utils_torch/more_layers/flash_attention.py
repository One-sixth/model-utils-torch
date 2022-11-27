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

    r = r1
    return r


if __name__ == '__main__':
    test_flash_attention()