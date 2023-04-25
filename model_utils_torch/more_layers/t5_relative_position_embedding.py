'''
google T5 的相对位置嵌入层
# Modify from huggingface transformer T5Attention
'''

import math
import torch
import torch.nn as nn


__all__ = ['T5_RelativePositionEmbedding']


@torch.jit.script
def _calc_rel_pos_bucket(rel_pos: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128):
    '''
    获得相对位置的桶的编号

    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        rel_pos:        long tensor - shape [query_L, key_L]
        bidirectional:  boolean     - whether the attention is bidirectional
        num_buckets:    integer
        max_distance:   integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    '''
    if bidirectional:
        num_buckets = num_buckets // 2
        rel_buckets = (rel_pos > 0).to(torch.long) * num_buckets
        rel_pos = torch.abs(rel_pos)
    else:
        rel_buckets = torch.zeros(1, dtype=torch.long, device=rel_pos.device)
        rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
    # now rel_pos is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = rel_pos < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    rel_pos_if_large = max_exact + \
                       (
                               torch.log(rel_pos.float() / max_exact)
                               / math.log(max_distance / max_exact)
                               * (num_buckets - max_exact)
                       ).long()
    rel_pos_if_large = torch.min(rel_pos_if_large, torch.full_like(rel_pos_if_large, num_buckets - 1))

    out = rel_buckets + torch.where(is_small, rel_pos, rel_pos_if_large)
    return out


class T5_RelativePositionEmbedding(torch.jit.ScriptModule):
    def __init__(self, emb_dim=16, num_buckets=32, max_distance=128, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.rel_pos_emb = nn.Embedding(self.num_buckets, emb_dim)

    @torch.jit.script_method
    def forward(self, query_L: int, key_L: int, rel_scale:int=1, rel_bias:int=0):
        '''Compute binned relative position bias'''
        device = self.rel_pos_emb.weight.device

        q_pos = torch.arange(query_L, dtype=torch.long, device=device)[:, None]
        k_pos = torch.arange(key_L, dtype=torch.long, device=device)[None, :]
        # 这里得到相对位置
        rel_pos = k_pos - q_pos  # shape (query_L, key_L)
        rel_pos = rel_pos * rel_scale + rel_bias

        # 通过相对位置获得桶的编号
        rel_pos_bucket = _calc_rel_pos_bucket(
            rel_pos,
            self.bidirectional,
            self.num_buckets,
            self.max_distance
        )

        # 获得嵌入向量
        emb = self.rel_pos_emb(rel_pos_bucket)  # shape (query_L, key_L, C)
        return emb


if __name__ == '__main__':
    m = T5_RelativePositionEmbedding()
    o = m(10, 20)
    print(o.shape)
    m = T5_RelativePositionEmbedding(bidirectional=False)
    o = m(10, 20)
    print(o.shape)
