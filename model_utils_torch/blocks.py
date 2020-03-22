'''
一些常见的残差块，和残差块堆叠类
这里不使用 torch.jit.ScriptModule
因为一些块可能会很复杂，所以不使用
另外，对于这种模块，更建议使用
torch.jit.script(mod)
来进行优化，这个需要 pytorch 1.3.1 或以上版本
'''

from .layers import *
from .temp_layers import *
from .ops import *


class BaseResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.act = act


class BaseDpnBlock(nn.Module):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, is_first):
        super().__init__()
        self.in_ch = in_ch
        self.ori_channels = ori_channels
        self.stride = stride
        self.act = act
        self.channels_increment = channels_increment
        self.is_first = is_first
        self.out_ch = in_ch + channels_increment


class ResBlock_1(BaseResBlock):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1 or in_ch != out_ch:
            self.shortcut = ConvBnAct2D(in_ch, out_ch, 3, stride, 1, nn.Identity())
        else:
            self.shortcut = nn.Identity()
        self.conv1 = ConvBnAct2D(in_ch, in_ch, 3, stride, 1, act)
        self.conv2 = ConvBnAct2D(in_ch, out_ch, 3, 1, 1, act)

    def forward(self, x):
        shortcut = self.shortcut(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = shortcut + y
        return y


class ResBlock_2(BaseResBlock):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1 or in_ch != out_ch:
            self.shortcut = ConvBnAct2D(in_ch, out_ch, 3, stride, 1, nn.Identity())
        else:
            self.shortcut = nn.Identity()
        self.conv1 = ConvBnAct2D(in_ch, in_ch*2, 1, 1, 0, act)
        self.conv2 = DwConvBnAct2D(in_ch*2, 1, 3, stride, 1, act)
        self.conv3 = ConvBnAct2D(in_ch*2, out_ch, 1, 1, 0, nn.Identity())

    def forward(self, x):
        shortcut = self.shortcut(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = shortcut + y
        return y


class DpnBlock_1(BaseDpnBlock):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, is_first, **kwargs):
        super().__init__(in_ch, ori_channels, stride, act, channels_increment, is_first)
        if np.max(stride) > 1 or is_first: # input_shape[1] != self.filters:
            self.shortcut = ConvBnAct2D(in_ch, ori_channels, 3, stride, 1, nn.Identity())
        else:
            self.shortcut = nn.Identity()
        self.conv1 = ConvBnAct2D(in_ch, ori_channels, 1, 1, 0, nn.Identity())
        self.conv2 = ConvBnAct2D(ori_channels, ori_channels, 3, stride, 1, act)
        self.conv3 = ConvBnAct2D(ori_channels, ori_channels+channels_increment, 1, 1, 0, nn.Identity())

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        res_outputs, dp_outputs = outputs.split([self.ori_channels, self.channels_increment], 1)
        shortcut_res, shortcut_dp = shortcut.split([self.ori_channels, self.in_ch-self.ori_channels], 1)
        shortcut_res = shortcut_res + res_outputs
        if int(shortcut_dp.shape[1]) == 0:
            shortcut_dp = dp_outputs
        else:
            shortcut_dp = torch.cat([shortcut_dp, dp_outputs], 1)
        outputs = torch.cat([shortcut_res, shortcut_dp], 1)
        return outputs


class DpnBlock_2(BaseDpnBlock):
    def __init__(self, in_ch, ori_channels, stride, act, channels_increment, is_first, **kwargs):
        super().__init__(in_ch, ori_channels, stride, act, channels_increment, is_first)
        if np.max(stride) > 1 or is_first:
            self.shortcut = ConvBnAct2D(in_ch, ori_channels, 3, stride, 1, nn.Identity())
        else:
            self.shortcut = nn.Identity()
        self.conv1 = ConvBnAct2D(in_ch, ori_channels, 1, 1, 0, nn.Identity())
        self.conv2 = DwConvBnAct2D(ori_channels, 1, 3, stride, 1, act)
        self.conv3 = ConvBnAct2D(ori_channels, ori_channels + channels_increment, 1, 1, 0, nn.Identity())

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)

        res_outputs, dp_outputs = outputs.split([self.ori_channels, self.channels_increment], 1)
        shortcut_res, shortcut_dp = shortcut.split([self.ori_channels, self.in_ch-self.ori_channels], 1)
        shortcut_res = shortcut_res + res_outputs
        if int(shortcut_dp.shape[1]) == 0:
            shortcut_dp = dp_outputs
        else:
            shortcut_dp = torch.cat([shortcut_dp, dp_outputs], 1)
        outputs = torch.cat([shortcut_res, shortcut_dp], 1)
        return outputs


class ResBlock_ShuffleNetV2(BaseResBlock):
    def __init__(self, in_ch, out_ch, stride, act, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        if np.max(stride) > 1:
            shortcut = DwConvBnAct2D(in_ch, 1, 3, stride, 1, nn.Identity())
            shortcut2 = ConvBnAct2D(in_ch, out_ch, 1, 1, 0, act)
            self.shortcut = nn.Sequential(shortcut, shortcut2)
            self.has_shortcut = True
        else:
            self.shortcut = nn.Identity()
            self.has_shortcut = False
            in_ch = in_ch // 2
            out_ch = out_ch // 2
        self.conv1 = ConvBnAct2D(in_ch, in_ch, 1, 1, 0, act)
        self.conv2 = DwConvBnAct2D(in_ch, 1, 3, stride, 1, nn.Identity())
        self.conv3 = ConvBnAct2D(in_ch, out_ch, 1, 1, 0, act)

    def forward(self, inputs):
        if self.has_shortcut:
            outputs1 = inputs
            outputs2 = self.shortcut(inputs)
        else:
            outputs1, outputs2 = inputs.chunk(2, 1)
        outputs1 = self.conv1(outputs1)
        outputs1 = self.conv2(outputs1)
        outputs1 = self.conv3(outputs1)
        if self.has_shortcut:
            outputs = outputs1 + outputs2
        else:
            outputs = torch.cat([outputs1, outputs2], 1)
        outputs = channel_shuffle(outputs, 4)
        return outputs


class GroupBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        mods = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=out_ch, out_ch=out_ch, stride=1, act=act, **kwargs)
            mods.append(block)
        self.mods = nn.ModuleList(mods)

    def forward(self, x):
        y = x
        for m in self.mods:
            y = m(y)
        return y


class GroupBlock_LastDown(nn.Module):
    '''与上面不同地方在于在最后一个残差块才下采样'''
    def __init__(self, in_ch, inter_channels, out_ch, stride, act, block_type, blocks, **kwargs):
        super().__init__()
        assert blocks >= 2, 'blocks must greate or equal than 2'
        self.in_ch = in_ch
        self.out_ch = out_ch
        mods = []
        for i in range(blocks):
            if i == 0:
                block = block_type(in_ch=in_ch, out_ch=inter_channels, stride=1, act=act, **kwargs)
            elif i == blocks-1:
                block = block_type(in_ch=inter_channels, out_ch=out_ch, stride=stride, act=act, **kwargs)
            else:
                block = block_type(in_ch=inter_channels, out_ch=inter_channels, stride=1, act=act, **kwargs)
            mods.append(block)
        self.mods = nn.ModuleList(mods)

    def forward(self, x):
        y = x
        for m in self.mods:
            y = m(y)
        return y


class GroupDpnBlock(nn.Module):
    def __init__(self, in_ch, stride, act, channels_increment, block_type, blocks, *, out_ch=None,
                 conv_setting=None, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        if out_ch:
            self.out_ch = out_ch
        else:
            self.out_ch = in_ch + channels_increment * blocks
        mods = []
        ori_channels = in_ch
        next_channels = in_ch
        for i in range(blocks):
            block = block_type(in_ch=next_channels, ori_channels=ori_channels,
                               channels_increment=channels_increment, stride=stride if i == 0 else 1, act=act,
                               is_first=i==0, **kwargs)
            next_channels += channels_increment
            mods.append(block)

        if out_ch:
            conv = ConvBnAct2D(in_ch + channels_increment * blocks, out_ch, **conv_setting)
            mods.append(conv)

        self.mods = nn.ModuleList(mods)

    def forward(self, x):
        y = x
        for m in self.mods:
            y = m(y)
        return y


class Hourglass4x(nn.Module):
    def __init__(self, in_ch, inter_ch, out_ch, act, block_type):
        super().__init__()

        self.ds = nn.AvgPool2d(3, 2, 1, True, False)

        self.gp_p1_head = GroupBlock_LastDown(in_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p2_head = GroupBlock_LastDown(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p3_head = GroupBlock_LastDown(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)
        self.gp_p4_head = GroupBlock_LastDown(inter_ch, inter_ch, inter_ch, 1, act, block_type, 3)

        self.gp_p0_body = GroupBlock_LastDown(in_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p1_body = GroupBlock_LastDown(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p2_body = GroupBlock_LastDown(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p3_body = GroupBlock_LastDown(inter_ch, inter_ch, out_ch, 1, act, block_type, 3)
        self.gp_p4_body = GroupBlock_LastDown(inter_ch, out_ch, out_ch, 1, act, block_type, 2)

        self.gp_p1_end = GroupBlock(out_ch, out_ch, 1, act, block_type, 1)
        self.gp_p2_end = GroupBlock(out_ch, out_ch, 1, act, block_type, 1)
        self.gp_p3_end = GroupBlock(out_ch, out_ch, 1, act, block_type, 1)

    def forward(self, x):
        skip0 = self.gp_p0_body(x)

        y = self.ds(x)
        y = self.gp_p1_head(y)
        skip1 = self.gp_p1_body(y)

        y = self.ds(y)
        y = self.gp_p2_head(y)
        skip2 = self.gp_p2_body(y)

        y = self.ds(y)
        y = self.gp_p3_head(y)
        skip3 = self.gp_p3_body(y)

        y = self.ds(y)
        y = self.gp_p4_head(y)
        y = self.gp_p4_body(y)
        y = resize_ref(y, skip3)
        y = y + skip3

        y = self.gp_p3_end(y)
        y = resize_ref(y, skip2)
        y = y + skip2

        y = self.gp_p2_end(y)
        y = resize_ref(y, skip1)
        y = y + skip1

        y = self.gp_p1_end(y)
        y = resize_ref(y, skip0)
        y = y + skip0

        return y


class Res2Block_1(BaseResBlock):
    def __init__(self, in_ch, out_ch, stride, act, scale=4, **kwargs):
        super().__init__(in_ch, out_ch, stride, act)
        assert np.max(stride) == 1, 'only support stride equal 1'
        assert out_ch % scale == 0, 'out_ch % scale must be equal 0'
        self.scale = scale
        self.conv1 = ConvBnAct2D(in_ch, out_ch, 1, 1, 0, act)
        self.conv2 = ConvBnAct2D(out_ch, out_ch, 1, 1, 0, act)
        self.conv_list = nn.ModuleList()
        inter_ch = out_ch // scale
        for _ in range(1, scale):
            self.conv_list.append(ConvBnAct2D(inter_ch, inter_ch, 3, stride, 1, act))

    def forward(self, x):
        y = self.conv1(x)
        ys = torch.chunk(y, self.scale, 1)
        ys2 = []
        for i in range(self.scale):
            if i == 0:
                ys2.append(ys[i])
            elif i == 1:
                y = self.conv_list[i-1](ys[i])
                ys2.append(y)
            else:
                y = ys2[i-1] + ys[i]
                y = self.conv_list[i-1](y)
                ys2.append(y)
        y = torch.cat(ys2, 1)
        y = self.conv2(y)
        return y
