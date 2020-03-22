import unittest
from model_utils_torch.temp_layers import *


class TestTempLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_1d = torch.randn(5, 10)
        self.test_data_2d = torch.randn(5, 10, 16, 16)

    def test_LinearBnAct(self):
        m = LinearBnAct(10, 20, nn.ReLU())
        y = m(self.test_data_1d)
        self.assertEqual((5, 20), tuple(y.shape))

    def test_ConvBnAct2D(self):
        m = ConvBnAct2D(10, 20, 3, 2, 1, nn.ReLU())
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 8, 8), tuple(y.shape))

    def test_DeConvBnAct2D(self):
        m = DeConvBnAct2D(10, 20, 2, 2, 0, nn.ReLU())
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 32, 32), tuple(y.shape))

    def test_DwConvBnAct2D(self):
        m = DwConvBnAct2D(10, 2, 3, 2, 1, nn.ReLU())
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 8, 8), tuple(y.shape))
