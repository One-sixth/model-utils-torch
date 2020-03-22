'''
v1.2
对模块进行重构

v1.1
过去的
'''

import torch as _torch
assert _torch.__version__ >= '1.3.1'

from . import acts
from . import ops
from . import layers
from . import blocks
from . import utils
from . import optim

from .acts import *
from .ops import *
from .layers import *
from .blocks import *
from .utils import *
from .optim import *
from . import image
from . import rev


__version__ = '1.2'
