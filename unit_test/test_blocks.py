import unittest
from model_utils_torch.blocks import *


class TestBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_2d = torch.randn(5, 10, 16, 16)
        self.test_data_2d_2 = torch.randn(5, 12, 16, 16)

    def test_ResBlock_1(self):
        act = nn.ReLU()
        m = ResBlock_1(10, 20, 2, act)
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 8, 8), tuple(y.shape))

        m = ResBlock_1(10, 20, 1, act)
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 16, 16), tuple(y.shape))

    def test_ResBlock_2(self):
        act = nn.ReLU()
        m = ResBlock_2(10, 20, 2, act)
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 8, 8), tuple(y.shape))

        m = ResBlock_2(10, 20, 1, act)
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 16, 16), tuple(y.shape))

    def test_DpnBlock_1(self):
        act = nn.ReLU()
        m1 = DpnBlock_1(10, 10, 2, act, 2, is_first=True)
        m2 = DpnBlock_1(12, 10, 1, act, 2, is_first=False)
        y = m1(self.test_data_2d)
        y = m2(y)
        self.assertEqual((5, 14, 8, 8), tuple(y.shape))

        m1 = DpnBlock_1(10, 10, 1, act, 2, is_first=True)
        m2 = DpnBlock_1(12, 10, 1, act, 2, is_first=False)
        y = m1(self.test_data_2d)
        y = m2(y)
        self.assertEqual((5, 14, 16, 16), tuple(y.shape))

    def test_DpnBlock_2(self):
        act = nn.ReLU()
        m1 = DpnBlock_2(10, 10, 2, act, 2, is_first=True)
        m2 = DpnBlock_2(12, 10, 1, act, 2, is_first=False)
        y = m1(self.test_data_2d)
        y = m2(y)
        self.assertEqual((5, 14, 8, 8), tuple(y.shape))

        m1 = DpnBlock_2(10, 10, 1, act, 2, is_first=True)
        m2 = DpnBlock_2(12, 10, 1, act, 2, is_first=False)
        y = m1(self.test_data_2d)
        y = m2(y)
        self.assertEqual((5, 14, 16, 16), tuple(y.shape))

    def test_ResBlock_ShuffleNetV2(self):
        act = nn.ReLU()
        m = ResBlock_ShuffleNetV2(12, 16, 2, act)
        y = m(self.test_data_2d_2)
        self.assertEqual((5, 16, 8, 8), tuple(y.shape))

        m = ResBlock_ShuffleNetV2(12, 12, 1, act)
        y = m(self.test_data_2d_2)
        self.assertEqual((5, 12, 16, 16), tuple(y.shape))

    def test_GroupBlock(self):
        act = nn.ReLU()
        m = GroupBlock(10, 16, 2, act, ResBlock_1, 2)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 8, 8), tuple(y.shape))

        m = GroupBlock(10, 16, 1, act, ResBlock_1, 2)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 16, 16), tuple(y.shape))

    def test_GroupBlock_LastDown(self):
        act = nn.ReLU()
        m = GroupBlock_LastDown(10, 10, 16, 2, act, ResBlock_1, 3)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 8, 8), tuple(y.shape))

        m = GroupBlock_LastDown(10, 10, 16, 1, act, ResBlock_1, 3)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 16, 16), tuple(y.shape))

    def test_GroupDpnBlock(self):
        act = nn.ReLU()
        m = GroupDpnBlock(10, 2, act, 2, DpnBlock_1, 3)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 8, 8), tuple(y.shape))

        m = GroupDpnBlock(10, 1, act, 2, DpnBlock_1, 3)
        y = m(self.test_data_2d)
        self.assertEqual((5, 16, 16, 16), tuple(y.shape))

    def test_Hourglass4x(self):
        act = nn.ReLU()
        m = Hourglass4x(10, 12, 14, act, ResBlock_1)
        y = m(self.test_data_2d)
        self.assertEqual((5, 14, 16, 16), tuple(y.shape))

    def test_Res2Block_1(self):
        act = nn.ReLU()

        m = Res2Block_1(10, 20, 1, act, scale=5)
        y = m(self.test_data_2d)
        self.assertEqual((5, 20, 16, 16), tuple(y.shape))
