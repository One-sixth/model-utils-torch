import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def circle_loss(feats: torch.Tensor, classes: torch.Tensor, scale=64, margin=0.25):
    '''
    CircleLoss
    ref https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py
    ref https://github.com/xialuxi/CircleLoss_Face/blob/master/CircleLoss.py
    ref https://github.com/zhen8838/Circle-Loss/blob/master/circle_loss.py
    老哥的代码写的真是好，极好的利用了各种向量化技巧，学习了
    :param feats:   输入特征，要求形状 [N, C]
    :param classes: 输入类别，这里类别号仅用来区分哪些是同类，哪些是不同类，要求形状 [N,]
    :param scale:   缩放参数，默认64
    :param margin:  边距参数，默认0.25
    :return:
    '''
    assert classes.ndim == 1
    assert feats.ndim == 2
    assert feats.shape[0] == classes.shape[0]

    B = classes.shape[0]
    # [B,] -> [B,1] -> [B,B]
    adj = classes[:, None].repeat(1, B)
    # 计算同类无向邻接矩阵，例如：例如第一行的第四个值为True，代表第一个样本和第四个样本是同类
    adj = adj == adj.T

    # 取上三角矩阵，但排除掉对角线。diagonal=0代表保留对角线的数值，而diagonal=1代表对角线往上一层，对角线的值不要了
    # 现在变成了同类配对的有向无环邻接矩阵
    pos_adj = adj.triu(diagonal=1)
    # 对mask取反后，再取上三角矩阵，然后和上面一样，对角线往上推移一层，对角线的值不要
    # 现在变成了异类配对的有向无环邻接矩阵
    neg_adj = (~adj).triu(diagonal=1)

    # 令特征向量长度为1。
    # 此时 cos 度量与 L2 度量等效，并且只需要简单乘法即可计算
    feats = F.normalize(feats, 2, 1)
    # 计算相似分数矩阵，越相似越接近1，否则越接近0
    sim_mat = torch.matmul(feats, feats.T)

    # 从相似分数矩阵里面获得同类样本对的相似分数
    pos_pair = sim_mat[pos_adj]
    # 从相似分数矩阵里面获得不同类样本对的相似分数
    neg_pair = sim_mat[neg_adj]

    # 后面就按公式计算了
    alpha_p = torch.relu(-pos_pair.detach() + 1 + margin)
    alpha_n = torch.relu(neg_pair.detach() + margin)
    margin_p = 1 - margin
    margin_n = margin

    # 这里会存在数值不稳定情况，已替换为下面的代码段
    # loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair - margin_p)))
    # loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair - margin_n)))
    # loss = torch.log(1 + loss_p * loss_n)

    # 避免了数值不稳定的问题
    logit_p = -alpha_p * (pos_pair - margin_p) * scale
    logit_n = alpha_n * (neg_pair - margin_n) * scale
    loss = F.softplus(logit_n.logsumexp(dim=0, keepdim=True) + logit_p.logsumexp(dim=0, keepdim=True))
    return loss


def generalized_dice_loss(preds, labels, class_weights=None, eps=1e-6):
    '''
    GeneralizedDiceLoss
    :param preds:           预测热图，要求值范围为[0,1]，形状[B,C,D1,D2,...]
    :param labels:          标签热图，要求值范围为[0,1]，形状[B,C,D1,D2,...]
    :param class_weights:   类别加权
    '''
    assert preds.shape == labels.shape
    assert 0 <= preds.data.min() <= preds.data.max() <= 1, 'Error! You need to ensure preds value range in [0, 1].'
    assert 0 <= labels.data.min() <= labels.data.max() <= 1, 'Error! You need to ensure labels value range in [0, 1].'

    label = labels
    pred = preds
    if class_weights is None:
        class_weights = [1] * label.shape[1]
    # ->[1,C]
    class_weights = torch.tensor(class_weights, dtype=label.dtype, device=label.device).reshape(1, label.shape[1])

    # label = torch.clip(label, 0, 1)
    # pred = torch.sigmoid(pred)

    # [B,C,D1,D2,...] -> [B,C,D1*D2*...]
    label = torch.flatten(label, 2)
    pred = torch.flatten(pred, 2)

    w = 1 / (label.sum([0, 2]) ** 2 + eps)
    # ->[1, C]
    w = w[None]

    # [B, C, L]->[B, C]
    inter = (label * pred).sum(2)
    bg = label.sum(2) + pred.sum(2)

    w_inter = w * inter
    w_bg = w * bg

    gen_dice = (2 * w_inter) / (w_bg + eps)

    gen_dice_loss = 1 - gen_dice
    gen_dice_loss = gen_dice_loss * class_weights

    gen_dice_loss = gen_dice_loss.mean()
    return gen_dice_loss


@torch.jit.script
def weighted_and_neg_topk_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    topk: Optional[int]=None,
    target_weight: Optional[torch.Tensor]=None,
    target_mask: Optional[torch.Tensor]=None,
    label_smoothing: float=0.,
    ignore_zero_target_weight: bool=True,
):
    '''
    加权的负权重topk交叉熵损失，主要用于NLG任务，对基于Sample的生成方式比较有效。
    主要行为：
    如果 目标项 对应的 target_weight 权重大于0，按照正常的交叉熵来计算
    如果 目标项 对应的 target_weight 权重小于0，先检查 目标项 的类别是否在 预测项前topk个高概率预测中，如果在，则按正常交叉熵来计算，如果不在，则跳过该项的计算。
    行为目的：
    忽略负向权重已掉出前topk的预测类别的梯度计算

    注意：
    如果使用了负向权重，在模型性能越好时，Loss值并非是单调下降的，可能会上升。并且Loss值可以小于0，然后Loss最小时并不是最优（训练最优）模型。如果需要用于评估，需要结合其他指标来评估。
    例如，可以使用多数为负值的 target_weight，可以发现 Loss 值是负的，然后在收敛后期时，Loss会反弹到0值处。

    以下维度缩写，B 代表批量大小，C 代表词向量维度
    虽然写着形状是 [B,C,...] 和 [B,...]
    :param input:                       FloatTensor shape [B,C,...] , 模型的输出
    :param target:                      LongTensor shape [B,...] , 预测目标
    :param topk:                        int or None , 检查前k个预测，None代表不使用，推荐使用10
    :param target_weight:               FloatTensor shape [B,...] or None , 每个目标的权值
    :param target_mask:                 BoolTensor shape [B,...] or None , 目标的掩码，True代表参与计算，False代表忽略
    :param label_smoothing:             float , 标签平滑
    :param ignore_zero_target_weight:   bool, 是否忽略 target_weight 中为0的目标，使其不参与梯度计算
    :return:
    '''
    assert target.shape[0] == input.shape[0] and target.shape[1:] == input.shape[2:], 'Error! Bad input and target shape.'
    assert topk is None or 0 < topk <= input.shape[1], 'Error! Bad param topk.'
    assert target_weight is None or target.shape == target_weight.shape, 'Error! Bad target_weight shape.'
    assert target_mask is None or (target.shape == target_mask.shape and target_mask.dtype == torch.bool), 'Error! Bad target_mask shape or dtype.'

    loss = F.cross_entropy(input, target, label_smoothing=label_smoothing, reduction='none')

    if target_weight is not None:
        loss = loss * target_weight

    if target_mask is None:
        target_mask = torch.full_like(target, 1, dtype=torch.bool)

    if target_weight is not None and topk is not None:
        # 如果负向权重的目标类别不在前K个列表中时，则跳过
        out_topk_cls = torch.topk(input.detach(), topk, dim=1, sorted=False)[1]
        # 筛选出 权重为负的，并且预测类别在前k个最高概率里的项
        neg_cls_slient_mask = torch.logical_and(~(target[:, None] == out_topk_cls).max(dim=1)[0], target_weight < 0)
        # 取反
        inv_neg_cls_slient_mask = ~neg_cls_slient_mask
        # 应用到 mask 上，即额外排除掉 权重为负的，并且预测类别不在前k个最高概率里的项
        target_mask = target_mask & inv_neg_cls_slient_mask

    if ignore_zero_target_weight and target_weight is not None:
        target_mask = target_mask & ~(target_weight == 0.)

    if target_mask.any().item():
        loss = loss[target_mask].mean()
    else:
        # 如果 mask 全部均为 False，代表 loss 为 0，为确保loss可以backward，所以使用 mul(0.) 处理
        loss = loss.sum().mul(0.)

    return loss


if __name__ == '__main__':
    batch_size = 6
    feats = torch.rand(batch_size, 16)
    classes = torch.randint(high=4, dtype=torch.long, size=(batch_size,))
    print(circle_loss(feats, classes).item())

    pred = torch.rand([3, 5, 10, 10])
    label = torch.rand([3, 5, 10, 10])
    print(generalized_dice_loss(pred, label).item())

    pred = torch.rand([1, 10])
    pred[0, 3] += 5
    label = torch.as_tensor([3], dtype=torch.long)
    topk = 9
    tgt_weight = torch.as_tensor([-0.5], dtype=torch.float32)
    tgt_mask = torch.as_tensor([True], dtype=torch.bool)

    print(weighted_and_neg_topk_cross_entropy(pred, label, topk, tgt_weight, tgt_mask).item())
