"""
Channel and Spatial CSNet Network (CS-Net).
"""
from __future__ import division
import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

def downsample():
    return nn.Pool(2, stride=2, op='maximum')

def deconv(in_channels, out_channels):
    return nn.ConvTranspose(in_channels, out_channels, 2, stride=2)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if (isinstance(m, nn.Conv) or isinstance(m, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if (m.bias is not None):
                    nn.init.constant_(m.bias, 0.0)
                    # m.bias.data.constant_()
            elif isinstance(m, nn.BatchNorm):
                m.weight.data.fill(1)
                nn.init.constant_(m.bias, 0.0)
                # m.bias.data.zero()

class ResEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm(out_channels)
        self.conv2 = nn.Conv(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv1x1 = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        residual = self.conv1x1(x)
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out += residual
        out = nn.relu(out)
        return out

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(nn.Conv(in_channels, out_channels, 3, padding=1), nn.BatchNorm(out_channels), nn.ReLU(), nn.Conv(out_channels, out_channels, 3, padding=1), nn.BatchNorm(out_channels), nn.ReLU())

    def execute(self, x):
        out = self.conv(x)
        return out

class SpatialAttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(nn.Conv(in_channels, (in_channels // 8), (1, 3), padding=(0, 1)), nn.BatchNorm((in_channels // 8)), nn.ReLU())
        self.key = nn.Sequential(nn.Conv(in_channels, (in_channels // 8), (3, 1), padding=(1, 0)), nn.BatchNorm((in_channels // 8)), nn.ReLU())
        self.value = nn.Conv(in_channels, in_channels, 1)
        self.gamma = jt.array(jt.zeros(1))
        self.softmax = nn.Softmax(dim=(- 1))

    def execute(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        (B, C, H, W) = x.shape
        proj_query = self.query(x).view(B, (- 1), (W * H)).permute((0, 2, 1))
        proj_key = self.key(x).view((B, (- 1), (W * H)))
        affinity = jt.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view((B, (- 1), (H * W)))
        weights = jt.matmul(proj_value, affinity.permute((0, 2, 1)))
        weights = weights.view((B, C, H, W))
        out = ((self.gamma * weights) + x)
        return out

class ChannelAttentionBlock(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = jt.array(jt.zeros(1))
        self.softmax = nn.Softmax(dim=(- 1))

    def execute(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        (B, C, H, W) = x.shape
        proj_query = x.view((B, C, (- 1)))
        proj_key = x.view(B, C, (- 1)).permute((0, 2, 1))
        affinity = jt.matmul(proj_query, proj_key)
        affinity_new = (jt.max(affinity, dim=-1, keepdims=True)[0].expand_as(affinity) - affinity)
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view((B, C, (- 1)))
        weights = jt.matmul(affinity_new, proj_value)
        weights = weights.view((B, C, H, W))
        out = ((self.gamma * weights) + x)
        return out

class AffinityAttention(nn.Module):
    ' Affinity attention module '

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)

    def execute(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = (sab + cab)
        return out

class CSNet(nn.Module):

    def __init__(self, classes, channels):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(CSNet, self).__init__()
        self.enc_input = ResEncoder(channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention(512)
        self.attention_fuse = nn.Conv((512 * 2), 512, 1)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv(32, classes, 1)
        initialize_weights(self)

    def execute(self, x):
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)
        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)
        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)
        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)
        input_feature = self.encoder4(down4)
        attention = self.affinity_attention(input_feature)
        attention_fuse = self.attention_fuse(jt.concat((input_feature, attention), dim=1))
        # attention_fuse = (input_feature + attention)
        up4 = self.deconv4(attention_fuse)
        up4 = jt.contrib.concat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)
        up3 = self.deconv3(dec4)
        up3 = jt.contrib.concat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)
        up2 = self.deconv2(dec3)
        up2 = jt.contrib.concat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)
        up1 = self.deconv1(dec2)
        up1 = jt.contrib.concat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)
        final = self.final(dec1)
        final = jt.sigmoid(final)
        return final