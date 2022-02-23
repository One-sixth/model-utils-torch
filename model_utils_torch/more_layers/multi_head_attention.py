'''
原始注意力层
'''

import torch
import torch.nn as nn
import math
from typing import Union

__all__ = ['MultiHeadAttention']


@torch.jit.script
def make_mask(mask: torch.Tensor, L: int):
    '''
    更加不同的输入情况，制作掩码
    支持 1D，2D，3D输入
    1D 输入时要求为long类型
    2D 和 3D 输入时，要求为bool类型
    [B, L1, L2]
    B 代表每句话的维度
    L1 代表每个字的维度
    L2 代表每个字对其他所有的字是否可见，1或True代表可见，0或False代表不可见
    :param mask:
    :param L:
    :return:
    '''
    if mask.ndim == 1:
        # mask [B]
        # 这个代表每批数据（每句话）用从头到末尾不同长度的掩码，每个字用相同的掩码
        assert mask.dtype == torch.long, 'Error! mask has wrong dtype.'
        m = torch.arange(L)[None,].repeat(mask.shape[0], 1)
        # m [B, L]
        m = m < mask[:, None]
        # m [B, 1, L]
        m = m[:, None, :]
    elif mask.ndim == 2:
        # mask [B, L]
        # 这个代表每批数据（每句话）用不同的掩码，但每个字都用同样的掩码
        assert mask.dtype == torch.bool, 'Error! mask has wrong dtype.'
        assert mask.shape[1] == L, 'Error! mask has bad shape.'
        # m [B, 1, L]
        m = mask[:, None, :]
    elif mask.ndim == 3:
        # mask [B, L, L]
        # 这个代表每批数据（每句话）的每个字都用不同的掩码
        assert mask.dtype == torch.bool, 'Error! mask has wrong dtype.'
        assert mask.shape[1] == mask.shape[2] == L, 'Error! mask has bad shape.'
        # m [B, L, L]
        m = mask
    else:
        raise AssertionError('Error! mask has bad shape.')

    m = torch.broadcast_to(m, [mask.shape[0], L, L])
    return m


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
    def forward(self, x, mask: Union[torch.Tensor, None]=None):
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

        # 应用mask
        if mask is not None:
            mask = make_mask(mask, L)
            # mask [B, L, L]
            assert B == mask.shape[0] and mask.shape[1] == mask.shape[2] == L, 'Error! mask has bad shape.'
            assert mask.dtype == torch.bool, 'Error! mask has wrong dtype.'
            attn = torch.where(mask[:, None], attn, torch.as_tensor(torch.inf, dtype=attn.dtype, device=attn.device))

        # 计算注意力分数
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
    # print(m.code)
    # print(make_mask.code)
    print(y.shape)

    r1 = y.shape == (3, 200, 128)

    mask1d = torch.randint(0, 200, [3,])
    mask2d = torch.randint(0, 200, [3, 200]) < 100
    mask3d = torch.randint(0, 200, [3, 200, 200]) < 100

    y1d = m(x, mask1d)
    y2d = m(x, mask2d)
    y3d = m(x, mask3d)
    print(y1d.shape)
    print(y2d.shape)
    print(y3d.shape)

    r2 = y1d.shape == y2d.shape == y3d.shape == (3, 200, 128)

    r = r1 and r2
    return r


if __name__ == '__main__':
    test_multi_head_attention()