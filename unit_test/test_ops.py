import unittest
from model_utils_torch.ops import *


class TestOps(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_2d = torch.randn(5, 10, 16, 16)

    def test_channel_shuffle(self):
        a = torch.arange(4).reshape(1, 4, 1, 1)
        y = channel_shuffle(a, 2).reshape(-1)
        self.assertEqual((0, 2, 1, 3), tuple(y))

    def test_resize_ref(self):
        a = torch.randn(5, 10, 32, 32)
        y = resize_ref(self.test_data_2d, a)
        self.assertEqual((5, 10, 32, 32), tuple(y.shape))

    def test_add_coord(self):
        y = add_coord(self.test_data_2d)
        self.assertEqual((5, 12, 16, 16), tuple(y.shape))

    def test_pixelwise_norm(self):
        y = pixelwise_norm(self.test_data_2d)
        self.assertEqual((5, 10, 16, 16), tuple(y.shape))

    def test_flatten(self):
        y = flatten(self.test_data_2d)
        self.assertEqual((5, 2560), tuple(y.shape))

    def test_adaptive_instance_normalization(self):
        style = torch.randn(5, 10, 1, 1)
        y = adaptive_instance_normalization(self.test_data_2d, style)
        self.assertEqual((5, 10, 16, 16), tuple(y.shape))

    def test_minibatch_stddev(self):
        y = minibatch_stddev(self.test_data_2d, 5, 1)
        self.assertEqual((5, 11, 16, 16), tuple(y.shape))

    def test_pixelshuffle(self):
        a = torch.randn(5, 16, 32, 32)
        y = pixelshuffle(a, (2, 2))
        y2 = F.pixel_shuffle(a, 2)
        b = torch.equal(y, y2)
        self.assertTrue(b)

    def test_pixelshuffle_invert(self):
        a = torch.randn(5, 16, 32, 32)
        x = F.pixel_shuffle(a, 2)
        y = pixelshuffle_invert(x, (2, 2))
        b = torch.equal(a, y)
        self.assertTrue(b)
