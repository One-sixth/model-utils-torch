'''
code from https://github.com/Serge-weihao/CCNet-Pure-Pytorch/blob/master/networks/CC.py
only mod code style

十字注意力模块
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['CrissCrossAttention']


class CrissCrossAttention(torch.jit.ScriptModule):
    '''
    十字注意力模块
    '''

    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    @torch.jit.script_method
    def forward(self, x):
        B, _, iH, iW = x.shape

        ninf_diag = torch.full([iH], fill_value=-torch.inf, device=x.device, dtype=x.dtype).diag(0)

        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).reshape(B * iW, -1, iH).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).reshape(B * iH, -1, iW).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).reshape(B * iW, -1, iH)
        proj_key_W = proj_key.permute(0, 2, 1, 3).reshape(B * iH, -1, iW)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).reshape(B * iW, -1, iH)
        proj_value_W = proj_value.permute(0, 2, 1, 3).reshape(B * iH, -1, iW)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + ninf_diag[None,].repeat(B * iW, 1, 1)) \
            .reshape(B, iW, iH, iH).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).reshape(B, iH, iW, iW)
        concate = F.softmax(torch.cat([energy_H, energy_W], 3), dim=3)
        att_H = concate[:, :, :, 0:iH].permute(0, 2, 1, 3).reshape(B * iW, iH, iH)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, iH:iH + iW].reshape(B * iH, iW, iW)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).reshape(B, iW, -1, iH).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).reshape(B, iH, -1, iW).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


if __name__ == '__main__':
    model = CrissCrossAttention(64)
    # x [B,C,H,W]
    x = torch.ones(10, 64, 32, 32)
    out = model(x)
    print(out.shape)
