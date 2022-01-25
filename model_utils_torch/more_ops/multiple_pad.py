'''
倍数填充算子
目前实现了 居中倍数填充

使用方法请看本文件内 test_center_multiple_pad 函数例子
'''

import torch
import torch.nn.functional as F
from typing import Union, List


__all__ = ['center_multiple_pad']


def calc_multiple_pad(dim: int, multiple: int):
    '''
    计算倍数填充的填充数
    :param dim:
    :param multiple:
    :return:
    '''
    pad = multiple - dim % multiple
    return pad


def calc_center_multiple_pad(dim: int, multiple: int):
    '''
    计算居中倍数填充的填充数
    :param dim:
    :param multiple:
    :return:
    '''
    pad = calc_multiple_pad(dim, multiple)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def center_multiple_pad(x: torch.Tensor, multiples: Union[List, int], pad_mode='constant', pad_value=0):
    '''
    倍数居中填充填充操作
    :param x:           输入张量
    :param multiples:   填充倍数，可以是整数或整数序列
    :param pad_mode:    填充方式
    :param pad_value:   填充值
    :return:

    multiples 可以是一个整数或整数序列

    举例
    multiples 为一个整数，举例BCHW，将会把HW维度填充为指定倍数
        x = torch.zeros(3,3,3,3)
        y = center_multi_pad(x, 5)
        y.shape == (3, 3, 5, 5)

    multiples 为一列整数序列时，举例BCHW，将会把HW维度填充为指定倍数，注意填充顺序是从最后一个维度开始的
        x = torch.zeros(3,3,3,3)
        y = center_multi_pad(x, [7, 8])
        y.shape == (3, 3, 8, 7)
        y = center_multi_pad(x, [5, 7, 8])
        y.shape == (3, 8, 7, 5)

    :return:
    '''
    multiples = [multiples] * (x.ndim - 2) if isinstance(multiples, int) else multiples
    assert len(multiples) <= x.ndim

    pads = []
    for m, dim in zip(multiples[::-1], x.shape[x.ndim - len(multiples):]):
        p1, p2 = calc_center_multiple_pad(dim, m)
        pads.append(p2)
        pads.append(p1)
    pads = pads[::-1]
    y = F.pad(x, pads, pad_mode, value=pad_value)
    return y


def test_center_multiple_pad():
    x = torch.zeros(3, 3, 3, 3)
    y = center_multiple_pad(x, 5)
    print(f'{(y.shape == (3, 3, 5, 5))=}')
    b1 = y.shape == (3, 3, 5, 5)

    x = torch.zeros(3, 3, 3, 3)
    y = center_multiple_pad(x, [7, 8])
    print(f'{(y.shape == (3, 3, 8, 7))=}')
    b1 &= y.shape == (3, 3, 8, 7)

    y = center_multiple_pad(x, [5, 7, 8])
    print(f'{(y.shape == (3, 8, 7, 5))=}')
    b1 &= y.shape == (3, 8, 7, 5)

    y = center_multiple_pad(x, [6, 5, 4], pad_mode='reflect')
    print(f'{(y.shape == (3, 4, 5, 6))=}')
    b1 &= y.shape == (3, 4, 5, 6)

    return b1


if __name__ == '__main__':
    test_center_multiple_pad()
