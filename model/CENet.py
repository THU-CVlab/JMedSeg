import jittor as jt
from jittor import init
from jittor import nn
from jittor.models.resnet import resnet34
from jittor.nn import relu, upsample, math

from functools import partial
nonlinearity = partial(relu)

class DACblock(nn.Module):

    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv(channel, channel, 3, dilation=3, padding=3)
        self.dilate3 = nn.Conv(channel, channel, 3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv(channel, channel, 1, dilation=1, padding=0)
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = ((((x + dilate1_out) + dilate2_out) + dilate3_out) + dilate4_out)
        return out

class DACblock_without_atrous(nn.Module):

    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.dilate3 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv(channel, channel, 1, dilation=1, padding=0)
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = ((((x + dilate1_out) + dilate2_out) + dilate3_out) + dilate4_out)
        return out

class DACblock_with_inception(nn.Module):

    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv(channel, channel, 1, dilation=1, padding=0)
        self.dilate3 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv((2 * channel), channel, 1, dilation=1, padding=0)
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(jt.contrib.concat([dilate1_out, dilate2_out], dim=1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = (x + dilate3_out)
        return out

class DACblock_with_inception_blocks(nn.Module):

    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv(channel, channel, 1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv(channel, channel, 5, dilation=1, padding=2)
        self.pooling = nn.Pool(3, stride=1, padding=1, op='maximum')
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)

    def execute(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = (((dilate1_out + dilate2_out) + dilate3_out) + dilate4_out)
        return out

class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv((features * (len(sizes) + 1)), out_features, 1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv(features, features, 1, bias=False)
        return nn.Sequential(prior, conv)

    def execute(self, feats):
        (h, w) = (feats.shape[2], feats.shape[3])
        priors = ([upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats])
        bottle = self.bottleneck(jt.contrib.concat(priors, dim=1))
        return nn.relu(bottle)

class SPPblock(nn.Module):

    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.Pool((2, 2), stride=2, op='maximum')
        self.pool2 = nn.Pool((3, 3), stride=3, op='maximum')
        self.pool3 = nn.Pool((5, 5), stride=5, op='maximum')
        self.pool4 = nn.Pool((6, 6), stride=6, op='maximum')
        self.conv = nn.Conv(in_channels, 1, 1, padding=0)

    def execute(self, x):
        (self.in_channels, h, w) = (x.shape[1], x.shape[2], x.shape[3])
        self.layer1 = upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')
        out = jt.contrib.concat([self.layer1, self.layer2, self.layer3, self.layer4, x], dim=1)
        return out

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv(in_channels, (in_channels // 4), 1)
        self.norm1 = nn.BatchNorm((in_channels // 4))
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose((in_channels // 4), (in_channels // 4), 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm((in_channels // 4))
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv((in_channels // 4), n_filters, 1)
        self.norm3 = nn.BatchNorm(n_filters)
        self.relu3 = nonlinearity

    def execute(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class CE_Net_(nn.Module):

    def __init__(self, num_classes=2, num_channels=3):
        super(CE_Net_, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 3, padding=1)

    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)
        d4 = (self.decoder4(e4) + e3)
        d3 = (self.decoder3(d4) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class CE_Net_backbone_DAC_without_atrous(nn.Module):

    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_DAC_without_atrous, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock_without_atrous(512)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 3, padding=1)

    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        d4 = (self.decoder4(e4) + e3)
        d3 = (self.decoder3(d4) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return jt.sigmoid(out)

class CE_Net_backbone_DAC_with_inception(nn.Module):

    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_DAC_with_inception, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock_with_inception(512)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 3, padding=1)

    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        d4 = (self.decoder4(e4) + e3)
        d3 = (self.decoder3(d4) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return jt.sigmoid(out)

class CE_Net_backbone_inception_blocks(nn.Module):

    def __init__(self, num_classes=1, num_channels=3):
        super(CE_Net_backbone_inception_blocks, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock_with_inception_blocks(512)
        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 3, padding=1)

    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        d4 = (self.decoder4(e4) + e3)
        d3 = (self.decoder3(d4) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return jt.sigmoid(out)

class CE_Net_OCT(nn.Module):

    def __init__(self, num_classes=12, num_channels=3):
        super(CE_Net_OCT, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = DACblock(512)
        self.spp = SPPblock(512)
        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 4, stride=2, padding=1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 3, padding=1)

    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.dblock(e4)
        e4 = self.spp(e4)
        d4 = (self.decoder4(e4) + e3)
        d3 = (self.decoder3(d4) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return out

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv(in_ch, out_ch, 3, padding=1), nn.BatchNorm(out_ch), nn.ReLU(), nn.Conv(out_ch, out_ch, 3, padding=1), nn.BatchNorm(out_ch), nn.ReLU())

    def execute(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def execute(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(nn.Pool(2, op='maximum'), double_conv(in_ch, out_ch))

    def execute(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose((in_ch // 2), (in_ch // 2), 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        diffX = (x1.shape[2] - x2.shape[2])
        diffY = (x1.shape[3] - x2.shape[3])
        x2 = F.pad(x2, ((diffX // 2), int((diffX / 2)), (diffY // 2), int((diffY / 2))))
        x = jt.contrib.concat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv(in_ch, out_ch, 1)

    def execute(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return jt.sigmoid(x)