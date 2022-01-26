'''
原始注意力层
'''

import torch
import torch.nn as nn
import math


__all__ = ['MultiHeadAttention']


class MultiHeadAttention(torch.jit.ScriptModule):
    __constants__ = ['head_dim', 'n_head']

    def __init__(self, in_dim, out_dim, head_dim, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim

        self.qkv_m = nn.Sequential(nn.Linear(in_dim, head_dim * n_head * 3, bias=True))

        if out_dim is None:
            self.out = nn.Identity()
        else:
            self.out = nn.Sequential(nn.Linear(head_dim * n_head, out_dim, bias=True))

    @torch.jit.script_method
    def forward(self, x):
        # x [B,L,C]
        B, L = x.shape[:2]
        qkv = self.qkv_m(x)
        qkv = qkv.reshape(B, L, 3, self.n_head, -1)
        # qkv [B,L,3,head,c]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv [3,B,head,L,c]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q,k,v [B,head,L,c]
        attn = q @ k.transpose(2, 3)
        # attn [B,head,L,L]
        # 需要先缩放注意力矩阵，否则性能会差很多
        attn = attn / math.sqrt(k.shape[-1])
        attn = attn.softmax(-1)

        y = attn @ v
        # y [B,head,L,c]
        y = y.transpose(1, 2)
        # y [B,L,head,c]
        y = y.reshape(B, L, -1)
        # y [B,L,C]
        y = self.out(y)
        return y


def test_multi_head_attention():
    x = torch.rand(3, 200, 128)
    # x [B, L, C]
    m = MultiHeadAttention(128, 128, 8, 8)
    y = m(x)
    print(y.shape)
    return y.shape == (3, 200, 128)


if __name__ == '__main__':
    test_multi_head_attention()