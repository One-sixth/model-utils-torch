import unittest
from model_utils_torch.utils import *


class TestUtils(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_2d = torch.randn(5, 10, 16, 16)

    def test_get_padding_by_name(self):
        self.assertEqual(1, get_padding_by_name(3, 'same'))
        self.assertEqual(2, get_padding_by_name(5, 'same'))
        self.assertEqual((1, 2), get_padding_by_name((3, 5), 'same'))
        self.assertEqual(0, get_padding_by_name(5, 'valid'))

    def test_fixup_init(self):
        w = torch.randn(20, 10, 3, 3)
        fixup_init(w, 3, 20)

    def test_print_params_size(self):
        m = nn.Conv2d(10, 20, 3, 1, 1)
        c = print_params_size(m)
        self.assertEqual(7280, c)

    def test__pair(self):
        pass

    def test_weight_clip_(self):
        m = nn.Conv2d(10, 20, 3, 1, 1)
        weight_clip_(m)
        weight_clip_(m.parameters())
