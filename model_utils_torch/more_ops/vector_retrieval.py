'''
向量检索算子
'''

import math
import torch
import torch.nn.functional as F


__all__ = ['find_closest_vector_by_L2', 'find_closest_vector_by_cos']


@torch.jit.script
def find_closest_vector_by_L2(in_vec: torch.Tensor, target_vec: torch.Tensor, group_size:int=64):
    '''
    在 target_vec 中搜索与 in_vec 最接近的向量，并返回最接近向量的编号
    L2距离法
    无梯度
    :param in_vec:      shape [B, C]，待检索向量组
    :param target_vec:  shape [L, C]，目标检索向量组
    :param group_size:  int，每组索引大小，避免爆显存
    :return: [B]，最接近向量的编号
    '''
    # 取消梯度跟踪
    in_vec = in_vec.data
    target_vec = target_vec.data

    B, C = in_vec.shape
    out_ids = []
    gs = group_size
    for bi in range(int(math.ceil(B / gs))):
        lens = torch.norm(in_vec[bi * gs:(bi + 1) * gs, None] - target_vec[None], 2, -1)
        # [B, len]
        ids = torch.argmin(lens, -1)
        out_ids.append(ids)
    out_ids = torch.cat(out_ids, 0)
    return out_ids


@torch.jit.script
def find_closest_vector_by_cos(in_vec: torch.Tensor, target_vec: torch.Tensor, group_size:int=64):
    '''
    在 target_vec 中搜索与 in_vec 最接近的向量，并返回最接近向量的编号
    余弦角度法
    无梯度
    :param in_vec:      shape [B, C]，待检索向量组
    :param target_vec:  shape [L, C]，目标检索向量组
    :param group_size:  int，每组索引大小，避免爆显存
    :return: [B]，最接近向量的编号
    '''
    # 取消梯度跟踪
    in_vec = in_vec.data
    target_vec = target_vec.data

    B, C = in_vec.shape
    out_ids = []
    gs = group_size
    for bi in range(int(math.ceil(B / gs))):
        cos = F.cosine_similarity(in_vec[bi * gs:(bi + 1) * gs, None], target_vec[None], -1)
        # [B, cos]
        ids = torch.argmax(cos, -1)
        out_ids.append(ids)
    out_ids = torch.cat(out_ids, 0)
    return out_ids


def test_find_closest_vector():
    in_vec = torch.as_tensor([
        [1, 1],
        [1, 0],
        [0, 1],
        [-1, -1],
    ], dtype=torch.float32)

    target_vec = torch.as_tensor([
        [1, 1],
        [1, 0],
        [1, -1],
        [0, 1],
        [0, -1],
        [-1, 1],
        [-1, 0],
        [-1, -1],
    ], dtype=torch.float32)

    out = find_closest_vector_by_L2(in_vec, target_vec)
    r1 = out.tolist() == [0, 1, 3, 7]

    out = find_closest_vector_by_cos(in_vec, target_vec)
    r2 = out.tolist() == [0, 1, 3, 7]

    return r1 and r2


if __name__ == '__main__':
    test_find_closest_vector()
