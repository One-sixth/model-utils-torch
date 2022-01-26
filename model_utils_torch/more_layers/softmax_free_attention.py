'''
代码来自 https://github.com/fudan-zvg/SOFT/blob/master/models/softmax_free_transformer.py
大幅度精简了代码

占用显存与原始注意力方法一致，效果与原始多头注意力层相似，但在大学习率1e-3情况下不稳定，验证集准确率受学习率影响较大
一般mode=0就行了
mode=1用于QKV长度不一致的情况，目前未考虑QKV长度不一致情况。
'''

import math
import torch
import torch.nn as nn


__all__ = ['SoftmaxFreeAttention']


@torch.jit.script
def subtraction_gaussian_kernel(q: torch.Tensor, k: torch.Tensor):
    # q 和 k 长度不一致
    device = q.device
    # [B, Head, L1, C] @ [C, L2] -> [B, Head, L1, L2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:], device=device)
    # [L1, C] @ [B, Head, C, L2] -> [B, Head, L1, L2]
    matB_square = torch.ones(q.shape[-2:], device=device) @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)


@torch.jit.script
def newton_inv(mat: torch.Tensor, max_iter: int):
    P = mat
    I = torch.eye(mat.shape[-1], device=mat.device)
    alpha = 2 / (torch.max(torch.sum(mat, dim=-1)) ** 2)
    beta = 0.5
    V = alpha * P
    pnorm = torch.max(torch.sum(torch.abs(I - P@V), dim=-2))
    err_cnt = 0
    while pnorm > 1.01 and err_cnt < 10:
        alpha *= beta
        V = alpha * P
        pnorm = torch.max(torch.sum(torch.abs(I - P@V), dim=-2))
        err_cnt += 1

    for i in range(max_iter):
        V = 2 * V - V @ P @ V
    return V


class SoftmaxFreeAttentionKernel(torch.jit.ScriptModule):
    __constants__ = ['head_dim', 'max_iter', 'mode']

    def __init__(self, head_dim: int, max_iter=20, mode=0):
        super().__init__()
        self.head_dim = head_dim
        self.max_iter = max_iter
        self.mode = mode

        self.Q_op = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim, bias=False),
            nn.LayerNorm(self.head_dim),
            nn.GELU())

    @torch.jit.script_method
    def forward(self, Q: torch.Tensor, V: torch.Tensor):
        B, n_head, L, head_dim, = Q.shape
        assert head_dim == self.head_dim

        Q = Q / math.pow(self.head_dim, 1 / 4)

        Q2 = self.Q_op(Q)
        # 这的K只是Q的转置，似乎可以换成单独生成一个K
        K2 = Q2.transpose(-1, -2)

        if self.mode == 0:
            # 如果 Q_op 没有让长度变短
            attn = subtraction_gaussian_kernel(Q2, K2)
            attn = torch.exp(-attn / 2)
            X = attn @ V
        else:
            # 如果 Q_op 让长度变短了
            ker1 = subtraction_gaussian_kernel(Q, K2)
            ker1 = torch.exp(-ker1 / 2)

            ker2 = subtraction_gaussian_kernel(Q2, K2)
            ker2 = torch.exp(-ker2 / 2)

            ker3 = ker1.transpose(-1, -2)

            X = (ker1 @ newton_inv(ker2, self.max_iter)) @ (ker3 @ V)

        return X


class SoftmaxFreeAttention(torch.jit.ScriptModule):
    __constants__ = ['in_dim', 'out_dim', 'n_head', 'head_dim', 'max_iter', 'mode']

    def __init__(self, in_dim, out_dim, head_dim, n_head, max_iter=20, mode=0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_iter = max_iter
        self.mode = mode

        self.W_q = nn.Linear(in_dim, n_head * head_dim)
        self.W_v = nn.Linear(in_dim, n_head * head_dim)

        self.attn_func = SoftmaxFreeAttentionKernel(head_dim, max_iter, mode=mode)

        if out_dim is None:
            self.out = nn.Identity()
        else:
            self.out = nn.Linear(n_head * head_dim, out_dim)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        # X [B, L, C]
        q = self.W_q(x)
        v = self.W_v(x)

        q = self.split_heads(q)
        v = self.split_heads(v)
        # [B, head, L, head_dim]

        attn_out = self.attn_func(q, v)
        attn_out = self.combine_heads(attn_out)

        out = self.out(attn_out)
        return out

    @torch.jit.script_method
    def combine_heads(self, x: torch.Tensor):
        # [B, head, L, head_dim]
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], self.n_head * self.head_dim)
        # [B, L, C]
        return x

    @torch.jit.script_method
    def split_heads(self, x: torch.Tensor):
        # [B, L, C]
        x = x.reshape(x.shape[0], x.shape[1], self.n_head, self.head_dim)
        x = x.transpose(1, 2)
        # [B, head, L, head_dim]
        return x


def test_softmax_free_attention():
    x = torch.rand(3, 200, 128).cuda()
    m = SoftmaxFreeAttention(128, 128, 8, 8).cuda()
    y = m(x)
    print(y.shape)
    return y.shape == (3, 200, 128)


if __name__ == '__main__':
    test_softmax_free_attention()
