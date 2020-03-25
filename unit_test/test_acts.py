import unittest
from model_utils_torch.acts import *


class TestActs(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fuzzy_check_nan(self, act):
        for _ in range(100):
            a = torch.rand(20, 100, requires_grad=True)
            x = (a - 1) * 1000
            y = act(x)
            y.sum().backward()
            self.assertTrue(torch.isnan(y).sum().item() == 0)
            self.assertTrue(torch.isnan(a.grad).sum().item() == 0)

    def test_LeakyTwiceRelu(self):
        self.fuzzy_check_nan(LeakyTwiceRelu())

    def test_TwiceLog(self):
        self.fuzzy_check_nan(TwiceLog())

    def test_TanhScale(self):
        self.fuzzy_check_nan(TanhScale())

    def test_Swish(self):
        self.fuzzy_check_nan(Swish())

    def test_SwishMemoryEfficient(self):
        self.fuzzy_check_nan(SwishMemoryEfficient())
