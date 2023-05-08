'''
Alter Cosine Scheduler
Modify from timm.scheduler.CosineLRScheduler

直接执行本文件，可以看到学习率变化曲线图
'''

import math
import numpy as np
import torch
from typing import Union
from timm.scheduler.scheduler import Scheduler


class AlterCosineLrScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr_min: float=1e-6,
                 cycle_base: float=4,
                 cycle_mul: float=2.,
                 cycle_recovery: float=1.,
                 cycle_limit: float=20.,
                 warmup_t=1.,
                 warmup_lr_init=1e-6,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True):
        '''
        修改的余弦学习率调整器
        注意，允许浮点数输入
        :param optimizer:       优化器输入
        :param lr_min:          最小学习率
        :param cycle_base:      起始周期大小
        :param cycle_mul:       周期增长倍数
        :param cycle_recovery:  回复周期轮数，如果不为0，学习率将会在周期末的最小学习率平滑恢复到下一个周期开始的最大学习率
        :param cycle_limit:     最大周期，到达最大周期后，不再继续增长
        :param warmup_t:        Warmup 周期数
        :param warmup_lr_init:  Warmup 起始学习率
        :param t_in_epochs:
        :param noise_range_t:
        :param noise_pct:
        :param noise_std:
        :param noise_seed:
        :param initialize:
        '''
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize
        )
        assert lr_min >= 0
        assert cycle_base >= 1
        assert cycle_mul > 1
        assert cycle_recovery >= 0
        assert cycle_limit > 1

        assert cycle_base > cycle_recovery
        assert cycle_limit >= cycle_base * cycle_mul

        step_times = [warmup_t]

        # 获得多少个中间周期
        n_cycle = cycle_limit / cycle_base / cycle_mul
        # 4*(2^0+2^1+2^2+2^3)

        if cycle_mul == 1:
            step_times.append(math.inf)

        else:
            for i in range(int(n_cycle)):
                cur_count = cycle_base * cycle_mul ** i
                if cur_count > cycle_limit:
                    break
                else:
                    cur_count += step_times[-1]
                step_times.append(cur_count)

        self.step_times = np.float32(step_times)

        self.lr_min = lr_min
        self.cycle_base = cycle_base
        self.cycle_mul = cycle_mul
        # self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.cycle_recovery = 1
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs

    def _get_lr(self, t):
        if t < self.step_times[0]:
            # warmup 过程
            factor = t / self.step_times[0]

        else:
            if t < self.step_times[-1]:
                # 周期增长过程
                i = np.argmax(t < self.step_times).item()
                cur_t = t

                seg_begin = self.step_times[i-1]
                seg_end = self.step_times[i]
                seg_mid = seg_end - self.cycle_recovery

            else:
                # 最大周期过程
                cur_t = t - self.step_times[-1]
                factor, base = math.modf(cur_t / self.cycle_limit)

                seg_begin = self.cycle_limit * base
                seg_end = self.cycle_limit * (base+1)
                seg_mid = seg_end - self.cycle_recovery

            # 计算小段
            if cur_t < seg_mid:
                factor = (cur_t - seg_begin) / (seg_mid - seg_begin)
            else:
                factor = 1 - (cur_t - seg_mid) / (seg_end - seg_mid)

        # 获得学习率调整因子
        cur_factor = (math.cos(math.pi * factor) + 1) / 2

        if t < self.step_times[0]:
            lrs = [self.warmup_lr_init * cur_factor + lr_base * (1-cur_factor) for lr_base in self.base_values]
        else:
            lrs = [self.lr_min * (1-cur_factor) + lr_base * cur_factor for lr_base in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: Union[int, float]):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = torch.zeros([1], requires_grad=True)

    optim = torch.optim.SGD([a], lr=1e-4)
    sch = AlterCosineLrScheduler(optim, lr_min=1e-5, cycle_base=4, cycle_mul=2, cycle_limit=20, warmup_t=2, warmup_lr_init=1e-6)

    xs = []
    ys = []

    for e in range(0, 100):
        if e == 120:
            print(1)
        for step in range(0, 200):
            x = e + step / 200
            sch.step(x)

            lr = optim.param_groups[0]['lr']

            xs.append(x)
            ys.append(lr)

    plt.figure(111)
    plt.plot(xs, ys, 'red')
    plt.xticks()
    plt.show()
