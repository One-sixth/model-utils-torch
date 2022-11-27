'''
v1.4.1
新算子 gen_sinusoidal_position_embedding，apply_rotary_position_embedding

v1.4
新层 FlashQuadSelfAttention，FlashQuadCrossAttention，T5_RelativePositionEmbedding
新算子 gen_nlp_self_attn_mask，gen_nlp_cross_attn_mask，find_closest_vector_by_L2，find_closest_vector_by_cos
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

from .acts import *
from .ops import *
from .layers import *
from .blocks import *
from .utils import *
from .optim import *
from .losses import *
from . import image
from . import rev


__version__ = '1.4'
