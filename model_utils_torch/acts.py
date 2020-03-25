import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F


class LeakyTwiceRelu(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        """
        """
        x = torch.where(x > 1, 1 + 0.1 * (x - 1), x)
        x = torch.where(x < 0, 0.1 * x, x)
        return x


class TwiceLog(torch.jit.ScriptModule):
    __constants__ = ['scale']

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    # 第一种实现
    # 不使用torch.sign，因为当x为0时，sign为0，此时梯度也为0
    # x为0时，torch.abs的梯度也为0，所以下面表达式不使用
    # sign = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, -1))
    # x = torch.log(torch.abs(x)+1) * sign
    # 第二种实现，当x=负数，而目标为正数时，梯度无效，原因，使用where后，图像是连接在一起，但导数函数仍然是分开的，例子 x=-1，x-3=0.7
    # x = torch.where(x >= 0, torch.log(x + 1), -1 * torch.log(torch.abs(x - 1)))
    # 第三种实现，当前实现，全程可导，而且导数域一致，忘记x本身就是线性可导了
    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        """
        """
        x = torch.where(x != 0, torch.log(torch.abs(x)+1) * torch.sign(x), x)
        x = x * self.scale
        return x


class TanhScale(torch.jit.ScriptModule):
    __constants__ = ['scale']

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        """
        """
        x = torch.tanh(x) * self.scale
        return x


# Copy from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py#L36.
class SwishMemoryEfficientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_outputs):
        x = ctx.saved_tensors[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_outputs * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class SwishMemoryEfficient(nn.Module):
    '''
    据说相比原始实现可以增加30%的批量大小，但不支持jit
    建议在训练时使用 SwishMemoryEfficient，导出jit模型时使用 Swish
    '''
    def forward(self, x):
        return SwishMemoryEfficientFunction.apply(x)


class Swish(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x * x.sigmoid()
