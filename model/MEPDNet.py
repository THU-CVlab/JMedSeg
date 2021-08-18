import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from jittorsummary import summary
from functools import reduce

np.random.seed(2020)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = ((kernel_size + 1) >> 1)
    if ((kernel_size % 2) == 1):
        center = (factor - 1)
    else:
        center = (factor - 0.5)
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = ((1 - (abs((og[0] - center)) / factor)) * (1 - (abs((og[1] - center)) / factor)))
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return jt.from_numpy(weight)

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, groups=1, last_relu=True):
        super().__init__()
        if (not mid_channels):
            mid_channels = out_channels
        modules = [nn.Conv(in_channels, mid_channels, 3, padding=1, groups=groups), nn.GroupNorm((mid_channels // 16), mid_channels, eps=1e-05, affine=None), nn.ReLU(), nn.Conv(mid_channels, out_channels, 3, padding=1, groups=groups), nn.GroupNorm((out_channels // 16), out_channels, eps=1e-05, affine=None)]
        if last_relu:
            modules.append(nn.ReLU())
        self.double_conv = nn.Sequential(*modules)

    def execute(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        modules = [nn.Conv(in_channels, out_channels, 3, padding=1, groups=groups), nn.GroupNorm(8, out_channels, eps=1e-05, affine=None)]
        if last_relu:
            modules.append(nn.ReLU())
        self.single_conv = nn.Sequential(*modules)

    def execute(self, x):
        return self.single_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.Pool(2, op='maximum'), DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu))

    def execute(self, x):
        return self.maxpool_conv(x)

class ReverseDown(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.down = nn.Pool(2, op='maximum')
        self.conv = DoubleConv((in_channels + (in_channels << 1)), out_channels, None, groups=groups, last_relu=last_relu)

    def execute(self, x1, x2):
        x1 = self.down(x1)
        x = jt.contrib.concat([x2, x1], dim=1)
        return self.conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv((in_channels + (in_channels >> 1)), out_channels, None, groups=groups, last_relu=last_relu)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        x = jt.contrib.concat([x2, x1], dim=1)
        return self.conv(x)

class UpAlone(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu)

    def execute(self, x):
        x = self.up(x)
        return self.conv(x)

class ReverseUp(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, last_relu=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, None, groups=groups, last_relu=last_relu)

    def execute(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)

class TopBlock(nn.Module):

    def __init__(self, channels):
        super(TopBlock, self).__init__()
        self.conv3 = nn.Sequential(nn.Conv(channels, channels, 3, padding=1, dilation=1), nn.GroupNorm((channels // 16), channels, eps=1e-05, affine=None), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv(channels, channels, 5, padding=2, dilation=1), nn.GroupNorm((channels // 16), channels, eps=1e-05, affine=None), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv(channels, channels, 7, padding=3, dilation=1), nn.GroupNorm((channels // 16), channels, eps=1e-05, affine=None), nn.ReLU())
        self.fuse = FuseBlock(channels, 3, post=False)

    def execute(self, x):
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        y = self.fuse(jt.stack((x1, x2, x3), dim=0))
        return y

class FuseBlock(nn.Module):

    def __init__(self, channels, branches, rate=16, min_length=32, post=False):
        super(FuseBlock, self).__init__()
        d = max((channels // rate), min_length)
        self.channels = channels
        self.branches = branches
        self.post = post
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(nn.Conv(channels, d, 1, bias=False), nn.GroupNorm((d // 16), d, eps=1e-05, affine=None), nn.ReLU())
        self.fc2 = nn.Conv(d, (channels * branches), 1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Sequential(nn.Conv(channels, channels, 1, stride=1, padding=0, bias=False), nn.GroupNorm((channels // 16), channels, eps=1e-05, affine=None))
        self.relu = nn.ReLU()

    def execute(self, inputs):
        batch = inputs[0].shape[0]
        s = self.pool(reduce((lambda x, y: (x + y)), inputs))
        ab = self.fc2(self.fc1(s))
        ab = self.softmax(ab.reshape(batch, self.branches, self.channels, -1))
        ab = map(lambda x: x.reshape(batch, self.channels, 1, 1), ab.chunk(self.branches, dim=1))
        V = map((lambda x, y: (x * y.expand_as(x))), inputs, ab)
        V = reduce((lambda x, y: (x + y)), V)
        if self.post:
            V = nn.relu((V + self.out(V)))
        return V

class VLambda(nn.Module):

    def __init__(self, channels, top=False, groups=8):
        super(VLambda, self).__init__()
        self.top = top
        self.down1 = Down(channels, (channels << 1), groups=groups)
        self.down2 = Down((channels << 1), (channels << 2), groups=groups)
        self.up1 = Up((channels << 2), (channels << 1), groups=groups)
        self.up2 = Up((channels << 1), channels, groups=groups)
        self.up3 = ReverseUp(channels, (channels >> 1), groups=groups)
        self.up4 = ReverseUp((channels >> 1), (channels >> 2), groups=groups)
        if self.top:
            self.topblock = TopBlock((channels >> 2))
        self.down3 = ReverseDown((channels >> 2), (channels >> 1), groups=groups)
        self.down4 = ReverseDown((channels >> 1), channels, groups=groups, last_relu=False)

    def execute(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.up1(x2, x1)
        x4 = self.up2(x3, x)
        x5 = self.up3(x4)
        x6 = self.up4(x5)
        if self.top:
            x6 = self.topblock(x6)
        x7 = self.down3(x6, x5)
        x8 = self.down4(x7, x)
        return nn.relu((x8 + x))

class MEPDNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(MEPDNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128, 8)
        self.down2 = Down(128, 256, 8)
        self.fuse = FuseBlock(256, 3, post=True)
        self.compression = nn.Sequential(nn.Conv(512, 256, 1, stride=1, padding=0, bias=False, groups=8), nn.GroupNorm((256 // 16), 256, eps=1e-05, affine=None), nn.ReLU())
        self.wave = VLambda(256, True, 8)
        self.up3 = UpAlone(256, 128, 8)
        self.up4 = UpAlone(128, 64, 8)
        self.outc = OutConv(64, n_classes)

    def execute(self, xu1, x, xd1):
        u3 = self.down2(self.down1(self.inc(xu1)))
        x3 = self.down2(self.down1(self.inc(x)))
        d3 = self.down2(self.down1(self.inc(xd1)))
        x = self.fuse(jt.stack((u3, x3, d3), dim=0))
        x = self.compression(jt.contrib.concat([x3, x], dim=1))
        x = self.wave(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits

def main():
    model = MEPDNet(n_channels=3, n_classes=2)
    x = jt.ones([2, 3, 512, 512])
    y = model(x,x,x)
    print (y.shape)
    _ = y.data

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    '''
    UNet
    17,276,290 total parameters.
    17,267,458 training parameters.
    '''

    from jittorsummary import summary
    summary(model, [(3,128,128), (3,128,128), (3,128,128)], device='cpu')

if __name__ == '__main__':
    main()