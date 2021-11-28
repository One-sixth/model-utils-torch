import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == '__main__':
    batch_size = 6
    feats = torch.rand(batch_size, 16)
    classes = torch.randint(high=4, dtype=torch.long, size=(batch_size,))
    print(circle_loss(feats, classes).item())

    pred = torch.rand([3, 5, 10, 10])
    label = torch.rand([3, 5, 10, 10])
    print(generalized_dice_loss(pred, label).item())
