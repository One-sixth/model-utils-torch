'''
v1.5
T5_RelativePositionEmbedding 允许调整的起始位置和缩放
增加新的损失函数 weighted_and_neg_topk_cross_entropy
新学习率调整器 AlterCosineLrScheduler
新函数 get_optim_cur_lr

破坏性更改
修复算子名称 gen_sinusoidal_position_embedding 到 make_sinusoidal_position_embedding
修复算子名称 gen_nlp_self_attn_mask，gen_nlp_cross_attn_mask 到 make_nlp_self_attn_mask，make_nlp_cross_attn_mask
增加 make_sinusoidal_position_embedding 参数，改变固定周期1000到可调节的周期，默认是10000

v1.4.1
新算子 make_sinusoidal_position_embedding，apply_rotary_position_embedding

v1.4
新层 FlashQuadSelfAttention，FlashQuadCrossAttention，T5_RelativePositionEmbedding
新算子 make_nlp_self_attn_mask，make_nlp_cross_attn_mask，find_closest_vector_by_L2，find_closest_vector_by_cos
新优化器 Adan

v1.3.1
引入losses包

v1.3
改变optim的结构
新增Adan优化器

v1.2
对模块进行重构

v1.1
过去的
'''

import torch
assert torch.__version__ >= '1.8.1'

from . import acts
from . import ops
from . import layers
from . import blocks
from . import utils
from . import optim
from . import losses
from . import scheduler

from .acts import *
from .ops import *
from .layers import *
from .blocks import *
from .utils import *
from .optim import *
from .losses import *
from .scheduler import *
from . import image
from . import rev


__version__ = '1.5'
