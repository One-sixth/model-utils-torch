import unittest

import torch
import math

from model_utils_torch.ops import *
from model_utils_torch.more_ops.multiple_pad import test_center_multiple_pad
from model_utils_torch.more_ops.vector_retrieval import test_find_closest_vector


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

    def test_one_hot(self):
        # dim == -1
        a = torch.randint(0, 100, [100, 100, 10])
        arr = one_hot(a, 100, dtype=torch.long)
        restruct_a = one_hot_invert(arr)
        b = torch.all(a == restruct_a)
        self.assertTrue(b)

        # dim != -1
        a = torch.randint(0, 100, [100, 100, 10])
        arr = one_hot(a, 100, dim=1, dtype=torch.long)
        restruct_a = one_hot_invert(arr, dim=1)
        b = torch.all(a == restruct_a)
        self.assertTrue(b)

    test_one_hot_invert = test_one_hot

    def test_multiple_pad(self):
        b = test_center_multiple_pad()
        self.assertTrue(b)

    def test_linspace_grid(self):
        ys = torch.linspace(-1, 1, 4)
        xs = torch.linspace(3, 5, 6)

        w = torch.meshgrid([ys, xs])
        w = torch.stack(w, 0)

        w2 = linspace_grid([4, 6], [-1, 1, 3, 5], 0)

        self.assertTrue(torch.allclose(w, w2))

    def test_vector_retrieval(self):
        self.assertTrue(test_find_closest_vector())

    def test_gen_sinusoidal_position_embedding(self):
        coord = make_sinusoidal_position_embedding(12, 0, 2, device='cpu')
        self.assertTrue(True)

    def test_apply_rotary_position_embedding(self):
        len = 12
        pos_ch = 6
        coord = make_sinusoidal_position_embedding(len, 0, pos_ch, device='cpu')
        x = torch.randn([2, len, pos_ch])
        out = apply_rotary_position_embedding(x, coord)
        self.assertTrue(True)
