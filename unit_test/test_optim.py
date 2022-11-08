import unittest
import torch
from model_utils_torch.optim import *


class TestOptim(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_adam16(self):
        p = torch.ones([1], dtype=torch.float16, requires_grad=True, device='cuda')
        optim = Adam16([p])
        p.sum().backward()
        optim.step()

    def test_adan(self):
        p = torch.ones([1], dtype=torch.float32, requires_grad=True)
        optim = Adan([p])
        p.sum().backward()
        optim.step()
