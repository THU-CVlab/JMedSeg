
import jittor as jt
from jittor import init
import os
from jittor import nn
import numpy as np

class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def execute(self, x):
        return x.view((x.data.shape[0], (- 1)))

class ConvLayer(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        self.add_module('conv', nn.Conv(in_channels, out_ch, kernel, stride=stride, padding=(kernel // 2), groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm(out_ch))
        self.add_module('relu', nn.ReLU6())

    def execute(self, x):
        return super().execute(x)

class HarDBlock(nn.Module):

    def get_link(self, layer, base_ch, growth_rate, grmul):
        if (layer == 0):
            return (base_ch, 0, [])
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = (2 ** i)
            if ((layer % dv) == 0):
                k = (layer - dv)
                link.append(k)
                if (i > 0):
                    out_channels *= grmul
        out_channels = (int((int((out_channels + 1)) / 2)) * 2)
        in_channels = 0
        for i in link:
            (ch, _, _) = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return (out_channels, in_channels, link)

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            (outch, inch, link) = self.get_link((i + 1), in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            layers_.append(ConvLayer(inch, outch))
            if (((i % 2) == 0) or (i == (n_layers - 1))):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def execute(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if (len(tin) > 1):
                x = jt.contrib.concat(tin, dim=1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (((i == 0) and self.keepBase) or (i == (t - 1)) or ((i % 2) == 1)):
                out_.append(layers_[i])
        out = jt.contrib.concat(out_, dim=1)
        return out

class HarDNet(nn.Module):

    def __init__(self, depth_wise=False, arch=85, pretrained=True, weight_path=''):
        super().__init__()
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]
        if (arch == 85):
            first_ch = [48, 96]
            ch_list = [192, 256, 320, 480, 720, 1280]
            gr = [24, 24, 28, 36, 48, 256]
            n_layers = [8, 16, 16, 16, 16, 4]
            downSamp = [1, 0, 1, 0, 1, 0]
            drop_rate = 0.2
        elif (arch == 39):
            first_ch = [24, 48]
            ch_list = [96, 320, 640, 1024]
            grmul = 1.6
            gr = [16, 20, 64, 160]
            n_layers = [4, 16, 8, 4]
            downSamp = [1, 1, 1, 0]
        if depth_wise:
            second_kernel = 1
            max_pool = False
            drop_rate = 0.05
        blks = len(n_layers)
        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))
        self.base.append(nn.Pool(3, stride=2, padding=1, op='maximum'))
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)
            if ((i == (blks - 1)) and (arch == 85)):
                self.base.append(nn.Dropout(0.1))
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if (downSamp[i] == 1):
                self.base.append(nn.Pool(2, stride=2, op='maximum'))
        ch = ch_list[(blks - 1)]
        # self.base.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Dropout(drop_rate), nn.Linear(ch, 1000)))

    def execute(self, x):
        out_branch = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            if ((i == 4) or (i == 9) or (i == 12) or (i == 15)):
                out_branch.append(x)
        out = x
        return out_branch

def hardnet(arch=68, pretrained=False, **kwargs):
    if (arch == 68):
        print('68 LOADED')
        model = HarDNet(arch=68)
    return model

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv(in_planes, out_planes, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(out_planes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)), BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)), BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3))
        self.branch2 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)), BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)), BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5))
        self.branch3 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1), BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)), BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)), BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7))
        self.conv_cat = BasicConv2d((4 * out_channel), out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def execute(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(jt.contrib.concat((x0, x1, x2, x3), dim=1))
        x = nn.relu((x_cat + self.conv_res(x)))
        return x

class aggregation(nn.Module):

    def __init__(self, channel, n_classes):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat2 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat3 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv4 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv5 = nn.Conv((3 * channel), n_classes, 1)

    def execute(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = (self.conv_upsample1(self.upsample(x1)) * x2)
        x3_1 = ((self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2))) * x3)
        x2_2 = jt.contrib.concat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = jt.contrib.concat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

class HarDMSEG(nn.Module):

    def __init__(self, channel=32, n_classes=2):
        super(HarDMSEG, self).__init__()
        self.relu = nn.ReLU()
        self.rfb2_1 = RFB_modified(320, channel)
        self.rfb3_1 = RFB_modified(640, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        self.agg1 = aggregation(channel, n_classes=n_classes)
        # self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        # self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        # self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        # self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        # self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        # self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # self.conv2 = BasicConv2d(320, 32, kernel_size=1)
        # self.conv3 = BasicConv2d(640, 32, kernel_size=1)
        # self.conv4 = BasicConv2d(1024, 32, kernel_size=1)
        # self.conv5 = BasicConv2d(1024, 1024, 3, padding=1)
        # self.conv6 = nn.Conv(1024, 1, 1)
        # self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hardnet = hardnet(arch=68)

    def execute(self, x):
        hardnetout = self.hardnet(x)
        x1 = hardnetout[0]
        x2 = hardnetout[1]
        x3 = hardnetout[2]
        x4 = hardnetout[3]
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = nn.interpolate(ra5_feat, scale_factor=8, mode='bilinear')
        return lateral_map_5

if __name__ == '__main__':
    test_model = HarDMSEG()
    x = jt.rand(10, 3, 512, 512)
    y = test_model(x)
    print(y.shape)