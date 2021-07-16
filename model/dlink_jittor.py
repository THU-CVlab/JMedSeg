import jittor as jt
from jittor import init
from jittor import nn
from jittor import models
from functools import partial
nonlinearity = partial(nn.relu)

class Dblock_more_dilate(nn.Module):

    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv(channel, channel, 3, dilation=2, padding=2)
        self.dilate3 = nn.Conv(channel, channel, 3, dilation=4, padding=4)
        self.dilate4 = nn.Conv(channel, channel, 3, dilation=8, padding=8)
        self.dilate5 = nn.Conv(channel, channel, 3, dilation=16, padding=16)
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    m.bias.data = jt.zeros(len(m.bias.data))

    def execute(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = (((((x + dilate1_out) + dilate2_out) + dilate3_out) + dilate4_out) + dilate5_out)
        return out

class Dblock(nn.Module):

    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv(channel, channel, 3, dilation=1, padding=1)
        self.dilate2 = nn.Conv(channel, channel, 3, dilation=2, padding=2)
        self.dilate3 = nn.Conv(channel, channel, 3, dilation=4, padding=4)
        self.dilate4 = nn.Conv(channel, channel, 3, dilation=8, padding=8)
        for m in self.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose)):
                if (m.bias is not None):
                    m.bias.data = jt.zeros(len(m.bias.data))

    def execute(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = ((((x + dilate1_out) + dilate2_out) + dilate3_out) + dilate4_out)
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

class DinkNet34_less_pool(nn.Module):

    def __init__(self, num_classes=1):
        super(DinkNet34_more_dilate, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.dblock = Dblock_more_dilate(256)
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
        e3 = self.dblock(e3)
        d3 = (self.decoder3(e3) + e2)
        d2 = (self.decoder2(d3) + e1)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        return F.sigmoid(out)

class DinkNet34(nn.Module):

    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = Dblock(512)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
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
        return F.sigmoid(out)

class LinkNet34(nn.Module):
    
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.finaldeconv1 = nn.ConvTranspose(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv(32, num_classes, 2, padding=1)
    
    def execute(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
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
class DinkNet50(nn.Module):

    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()
        filters = [256, 512, 1024, 2048]
        resnet = models.resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dblock = Dblock_more_dilate(2048)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
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



