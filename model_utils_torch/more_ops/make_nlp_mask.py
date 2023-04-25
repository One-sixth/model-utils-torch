'''
生成文本任务所需的注意力掩码
'''

import torch
import torch.nn.functional as F


@torch.jit.script
def make_nlp_self_attn_mask(mask:torch.Tensor, bidirectional:bool):
    '''
    生成自注意力掩码
    :param mask:            shape [B, L]
    :param bidirectional:   是否双向
    :return:
    '''
    mask = mask[:, None, :] * mask[:, :, None]
    # mask [B, L, L]
    if not bidirectional:
        mask *= torch.tril(torch.ones_like(mask), 0)
    return mask


@torch.jit.script
def make_nlp_cross_attn_mask(query_mask:torch.Tensor, key_mask:torch.Tensor, bidirectional:bool):
    '''
    生成交叉注意力掩码
    :param query_mask:      shape [B, L1]
    :param key_mask:        shape [B, L2]
    :param bidirectional:   是否双向
    :return:
    '''
    # query_mask [B, L1]
    # key_mask [B, L2]
    # [B, L1, 1] x [B, 1, L2]
    mask = query_mask[:,:,None] * key_mask[:,None,:]
    # mask [B, L1, L2]
    if not bidirectional:
        mask *= torch.tril(torch.ones_like(mask), 0)
    return mask


if __name__ == '__main__':
    q_mask = torch.as_tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ], dtype=torch.bool)

    k_mask = torch.as_tensor([
        [1, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
    ], dtype=torch.bool)

    self_attn_mask_1 = make_nlp_self_attn_mask(q_mask, True)
    self_attn_mask_2 = make_nlp_self_attn_mask(q_mask, False)
    cross_attn_mask_1 = make_nlp_cross_attn_mask(q_mask, k_mask, True)
    cross_attn_mask_2 = make_nlp_cross_attn_mask(q_mask, k_mask, False)
