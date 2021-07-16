import jittor as jt
from jittor import init
import math
from os.path import join as pjoin
from collections import OrderedDict
from jittor import nn

def np2th(weights, conv=False):
    'Possibly convert HWIO to OIHW.'
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
        
    return jt.float32(weights)

class StdConv2d(nn.Conv):

    def execute(self, x):
        w = self.weight
        # (v, m) = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        m = jt.mean(w, dims=(1, 2, 3), keepdims=True)
        v = jt.mean((w - m) ** 2, dims=(1,2,3), keepdims=True)
        w = ((w - m) / jt.sqrt((v + 1e-05)))
        return nn.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)

def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)

class PreActBottleneck(nn.Module):
    'Pre-activation (v2) bottleneck block.\n    '

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = (cout or cin)
        cmid = (cmid or (cout // 4))
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-06, affine=None)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-06, affine=None)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-06, affine=None)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU()
        if ((stride != 1) or (cin != cout)):
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout, affine=None)

    def execute(self, x):
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = nn.relu(self.gn1(self.conv1(x)))
        y = nn.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        y = nn.relu((residual + y))
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, 'conv1/kernel')], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, 'conv2/kernel')], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, 'conv3/kernel')], conv=True)
        gn1_weight = np2th(weights[pjoin(n_block, n_unit, 'gn1/scale')])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, 'gn1/bias')])
        gn2_weight = np2th(weights[pjoin(n_block, n_unit, 'gn2/scale')])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, 'gn2/bias')])
        gn3_weight = np2th(weights[pjoin(n_block, n_unit, 'gn3/scale')])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, 'gn3/bias')])
        # self.conv1.weight.copy_(conv1_weight)
        # self.conv2.weight.copy_(conv2_weight)
        # self.conv3.weight.copy_(conv3_weight)
        # self.gn1.weight.copy_(gn1_weight.view((- 1)))
        # self.gn1.bias.copy_(gn1_bias.view((- 1)))
        # self.gn2.weight.copy_(gn2_weight.view((- 1)))
        # self.gn2.bias.copy_(gn2_bias.view((- 1)))
        # self.gn3.weight.copy_(gn3_weight.view((- 1)))
        # self.gn3.bias.copy_(gn3_bias.view((- 1)))
        self.conv1.weight = (conv1_weight)
        self.conv2.weight = (conv2_weight)
        self.conv3.weight = (conv3_weight)
        self.gn1.weight = (gn1_weight.view((- 1)))
        self.gn1.bias = (gn1_bias.view((- 1)))
        self.gn2.weight = (gn2_weight.view((- 1)))
        self.gn2.bias = (gn2_bias.view((- 1)))
        self.gn3.weight = (gn3_weight.view((- 1)))
        self.gn3.bias = (gn3_bias.view((- 1)))
        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, 'conv_proj/kernel')], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, 'gn_proj/scale')])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, 'gn_proj/bias')])
            # self.downsample.weight.copy_(proj_conv_weight)
            # self.gn_proj.weight.copy_(proj_gn_weight.view((- 1)))
            # self.gn_proj.bias.copy_(proj_gn_bias.view((- 1)))
            self.downsample.weight = (proj_conv_weight)
            self.gn_proj.weight = (proj_gn_weight.view((- 1)))
            self.gn_proj.bias = (proj_gn_bias.view((- 1)))

class ResNetV2(nn.Module):
    'Implementation of Pre-activation (v2) ResNet mode.'

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int((64 * width_factor))
        self.width = width
        self.root = nn.Sequential(OrderedDict([('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)), ('gn', nn.GroupNorm(32, width, eps=1e-06, affine=None)), ('relu', nn.ReLU())]))
        self.body = nn.Sequential(OrderedDict([('block1', nn.Sequential(OrderedDict(([('unit1', PreActBottleneck(cin=width, cout=(width * 4), cmid=width))] + [(f'unit{i:d}', PreActBottleneck(cin=(width * 4), cout=(width * 4), cmid=width)) for i in range(2, (block_units[0] + 1))])))), ('block2', nn.Sequential(OrderedDict(([('unit1', PreActBottleneck(cin=(width * 4), cout=(width * 8), cmid=(width * 2), stride=2))] + [(f'unit{i:d}', PreActBottleneck(cin=(width * 8), cout=(width * 8), cmid=(width * 2))) for i in range(2, (block_units[1] + 1))])))), ('block3', nn.Sequential(OrderedDict(([('unit1', PreActBottleneck(cin=(width * 8), cout=(width * 16), cmid=(width * 4), stride=2))] + [(f'unit{i:d}', PreActBottleneck(cin=(width * 16), cout=(width * 16), cmid=(width * 4))) for i in range(2, (block_units[2] + 1))]))))]))

    def execute(self, x):
        features = []
        (b, c, in_size, _) = x.shape
        x = self.root(x)
        features.append(x)
        x = nn.Pool(3, stride=2, padding=0, op='maximum')(x)
        for i in range((len(self.body) - 1)):
            x = self.body[i](x)
            right_size = int(((in_size / 4) / (i + 1)))
            if (x.shape[2] != right_size):
                pad = (right_size - x.shape[2])
                assert ((pad < 3) and (pad > 0)), 'x {} should {}'.format(x.shape, right_size)
                # feat = jt.zeros((b, x.shape[1], right_size, right_size), device=x.device)
                feat = jt.zeros((b, x.shape[1], right_size, right_size))
                feat[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[(- 1)](x)
        return (x, features[::(- 1)])