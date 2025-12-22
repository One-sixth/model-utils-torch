'''
浮点值转向量功能

用浮点值在基数数组中寻找坐标，获得相对偏移权重，然后从向量集里选取向量
从而实现类似 软分类嵌入的效果
'''

import torch


@torch.jit.script
def compute_val2vec_weight(x: torch.Tensor, base_arr: torch.Tensor):
    """
    计算值在基数数组中的位置权重
    """
    assert isinstance(x, torch.Tensor), "输入x必须是一个tensor"
    assert x.ndim >= 1, "输入x至少有1维"
    assert isinstance(base_arr, torch.Tensor), "输入base_arr必须一个tensor"
    assert base_arr.ndim == 1, "输入base_arr必须是一维的"

    N = base_arr.shape[0]

    ori_x_shape = x.shape
    x = x.flatten()

    M = x.shape[0]

    # 查找插入位置
    indices = torch.searchsorted(base_arr, x, right=True)

    # 处理边界
    left_idx = torch.clamp(indices - 1, 0, N - 1)
    right_idx = torch.clamp(indices, 0, N - 1)

    # 获取对应的基数值
    left_vals = base_arr[left_idx]
    right_vals = base_arr[right_idx]

    # 计算权重
    mask_same = left_idx == right_idx
    mask_diff = ~mask_same

    # 初始化权重
    weights = torch.zeros([M, N], dtype=torch.float32, device=x.device)

    indices = torch.arange(M, device=x.device)

    # 处理相同索引的情况（边界或精确匹配）
    if mask_same.any():
        tgt_indices = indices[mask_same]
        weights[tgt_indices, left_idx[mask_same]] = 1.0

    # 处理不同索引的情况（需要插值）
    if mask_diff.any():
        tgt_indices = indices[mask_diff]
        l_idx = left_idx[mask_diff]
        r_idx = right_idx[mask_diff]

        # 计算插值权重
        a = left_vals[mask_diff]
        b = right_vals[mask_diff]
        x_vals = x[mask_diff]

        w_right = (x_vals - a) / (b - a + 1e-8)  # 添加小量避免除以0
        w_left = 1 - w_right

        # 分配权重
        weights[tgt_indices, l_idx] = w_left
        weights[tgt_indices, r_idx] = w_right

    # 恢复输入的形状
    weights = weights.unflatten(0, ori_x_shape)
    # left_idx = left_idx.unflatten(0, ori_x_shape)
    # right_idx = right_idx.unflatten(0, ori_x_shape)

    # return weights, left_idx, right_idx
    return weights


@torch.jit.script
def compute_val2vec(x: torch.Tensor, base_arr: torch.Tensor, vectors: torch.Tensor):
    '''
    浮点值根据基数数组获得相对位置权重，然后使用权重选取向量

    :param x:           浮点张量，不可以求导，形状是 [N1, ..., N2]
    :param base_arr:    基数数组，不可以求导，形状是 [M]
    :param vectors:     选取的向量，可以求导，向量数量必须等于基数数量，仅限选中的1个或两个才有导数，形状是 [M, C]
    :return: 选取出来的向量融合，形状是 [N1, ..., N2, C]
    '''
    assert base_arr.ndim == 1, "输入arr必须为1维张量"
    assert vectors.ndim == 2, "输入vectors必须为二维张量"
    assert vectors.shape[0] == base_arr.shape[0], "输入vectors和base_arr的维度不匹配"

    # weights, left_idx, right_idx = compute_val2vec_weight(x, base_arr)
    weights = compute_val2vec_weight(x, base_arr)

    # weight 需要压扁除了最后一个维度
    weights = weights.flatten(0, -2)
    # 向量化计算
    out = weights @ vectors
    # 恢复0维到输入的形状
    out = out.unflatten(0, x.shape)

    return out, weights


def test_compute_val2vec():
    print("\n=== val2vec 测试 ===")

    # 基数数组
    base_arr = torch.tensor([10, 50, 100, 1000, 1500, 2500], dtype=torch.float32)
    # 随机向量
    vectors = torch.randn([6, 8])  # 5个8维向量

    # 批处理输入
    batch_x = torch.tensor([-5, 5, 12, 70, 400, 500, 2400, 3000])

    selected_vectors, weights = compute_val2vec(batch_x, base_arr, vectors)

    print(f"批处理输入: {batch_x.tolist()}")
    print(f"权重矩阵形状: {weights.shape}")
    print(f"选择的向量形状: {selected_vectors.shape}")

    # 显示每个样本的权重分布
    for i, x in enumerate(batch_x):
        non_zero_idx = torch.nonzero(weights[i]).squeeze()
        non_zero_vals = weights[i, non_zero_idx]
        print(f"\n样本 {i}, x={x:.1f}:")
        print(f"  非零权重位置: {non_zero_idx.tolist()}")
        print(f"  权重值: {non_zero_vals.tolist()}")


if __name__ == "__main__":
    test_compute_val2vec()
