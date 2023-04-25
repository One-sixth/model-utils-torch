import unittest

import torch
import math

from model_utils_torch.losses import *


class TestLosses(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_circle_loss(self):
        batch_size = 6
        feats = torch.rand(batch_size, 16)
        classes = torch.randint(high=4, dtype=torch.long, size=(batch_size,))
        self.assertTrue(circle_loss(feats, classes, 64, 0.25).item() >= 0)

    def test_generalized_dice_loss(self):
        pred = torch.rand([3, 5, 10, 10])
        label = torch.rand([3, 5, 10, 10])
        self.assertTrue(generalized_dice_loss(pred, label).item() >= 0)

    def test_weighted_and_neg_topk_cross_entropy(self):
        pred = torch.rand([1, 10])
        pred[0, 3] += 5
        label = torch.as_tensor([3], dtype=torch.long)
        topk = 9
        tgt_weight = torch.as_tensor([-0.5], dtype=torch.float32)
        tgt_mask = torch.as_tensor([True], dtype=torch.bool)

        loss = weighted_and_neg_topk_cross_entropy(pred, label, topk, tgt_weight, tgt_mask).item()
        self.assertTrue(True)
