import unittest
from model_utils_torch import *
from model_utils_torch.more_layers.softmax_free_attention import test_softmax_free_attention
from model_utils_torch.more_layers.multi_head_attention import test_multi_head_attention
from model_utils_torch.more_layers.flash_attention import test_flash_attention


class TestLayers(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_1d = torch.randn(5, 10)
        self.test_data_2d = torch.randn(5, 10, 16, 16)
        self.test_data_3d = torch.randn(5, 10, 16, 16, 16)

    def test_Upsample(self):
        m = Upsample(size=(32, 32), mode='bilinear', align_corners=False)
        y = m(self.test_data_2d)
        self.assertEqual((5, 10, 32, 32), tuple(y.shape))

        m = Upsample(scale_factor=2, mode='area', align_corners=None)
        y = m(self.test_data_2d)
        self.assertEqual((5, 10, 32, 32), tuple(y.shape))

    def test_UpsampleConcat(self):
        m = UpsampleConcat('bilinear', align_corners=False)
        a = torch.randn(5, 10, 32, 32)
        y = m(self.test_data_2d, a)
        self.assertEqual((5, 20, 32, 32), tuple(y.shape))

    def test_LinearGroup(self):
        m = LinearGroup(10, 20, 2)
        y = m(self.test_data_1d)
        self.assertEqual((5, 20), tuple(y.shape))

        m = LinearGroup(10, 20, 2, bias=False)
        y = m(self.test_data_1d)
        self.assertEqual((5, 20), tuple(y.shape))

    def test_CrissCrossAttention(self):
        m = CrissCrossAttention(10)
        y = m(self.test_data_2d)
        self.assertEqual((5, 10, 16, 16), tuple(y.shape))

    def test_SwitchableNormND(self):
        m = SwitchableNormND(2, 10)
        y = m(self.test_data_2d)
        self.assertEqual((5, 10, 16, 16), tuple(y.shape))

        m = SwitchableNorm(10)
        y = m(self.test_data_1d)
        self.assertEqual((5, 10), tuple(y.shape))

        m = SwitchableNorm1D(10)
        y = m(self.test_data_1d[..., None])
        self.assertEqual((5, 10, 1), tuple(y.shape))

        m = SwitchableNorm2D(10)
        y = m(self.test_data_2d)
        self.assertEqual((5, 10, 16, 16), tuple(y.shape))

        m = SwitchableNorm3D(10)
        y = m(self.test_data_3d)
        self.assertEqual((5, 10, 16, 16, 16), tuple(y.shape))

    def test_softmax_free_attention(self):
        b = test_softmax_free_attention()
        self.assertTrue(b)

    def test_multi_head_attention(self):
        b = test_multi_head_attention()
        self.assertTrue(b)

    def test_flash_attention(self):
        b = test_flash_attention()
        self.assertTrue(b)

    def test_flash_attention_2(self):
        # shape [B,L,C]
        q = torch.rand([2, 4, 12])
        k = torch.rand([2, 6, 12])

        q_mask = torch.as_tensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ], dtype=torch.bool)
        k_mask = torch.as_tensor([
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ], dtype=torch.bool)

        k_self_mask = gen_nlp_self_attn_mask(k_mask, True)
        qk_cross_mask = gen_nlp_cross_attn_mask(q_mask, k_mask, False)

        k_self_mask = torch.where(k_self_mask, 0, -torch.inf)
        qk_cross_mask = torch.where(qk_cross_mask, 0, -torch.inf)

        self_attn = FlashQuadSelfAttention(12, 12, 24, 8)
        cross_attn = FlashQuadCrossAttention(12, 12, 24, 8)

        k = self_attn(k, attn_bias=k_self_mask)
        y = cross_attn(q, k, attn_bias=qk_cross_mask)

        # print(k.shape)
        # print(y.shape)

        self.assertTrue(True)

    def test_t5_relative_position_embedding(self):
        m = T5_RelativePositionEmbedding(bidirectional=True)
        o = m(10, 20)
        m = T5_RelativePositionEmbedding(bidirectional=False)
        o = m(10, 20)
        self.assertTrue(True)
