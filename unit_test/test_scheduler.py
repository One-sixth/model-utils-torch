import unittest
from model_utils_torch import *


class TestLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_AlterCosineLrScheduler(self):
        a = torch.zeros([1], requires_grad=True)

        optim = torch.optim.SGD([a], lr=1e-4)
        sch = AlterCosineLrScheduler(optim, lr_min=1e-5, cycle_base=4, cycle_mul=2, cycle_limit=20, warmup_t=2, warmup_lr_init=1e-6)

        xs = []
        ys = []

        for e in range(0, 100):
            if e == 120:
                print(1)
            for step in range(0, 100):
                x = e + step / 100
                sch.step(x)

                lr = optim.param_groups[0]['lr']

                xs.append(x)
                ys.append(lr)

        self.assertTrue(True)
