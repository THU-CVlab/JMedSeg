import jittor as jt
from jittor import init
from jittor import nn

from jittor.models.vgg import vgg16

def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv(in_, out, 3, padding=1)

class ConvRelu(nn.Module):

    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), nn.ConvTranspose(middle_channels, out_channels, 3, stride=2, padding=1, output_padding=1), nn.ReLU())

    def execute(self, x):
        return self.block(x)

class UNet11(nn.Module):

    def __init__(self, num_filters: int=32, pretrained: bool=False) -> None:
        '\n        Args:\n            num_filters:\n            pretrained:\n                False - no pre-trained network is used\n                True  - encoder is pre-trained with VGG11\n        '
        super().__init__()
        self.pool = nn.Pool(2, stride=2, op='maximum')
        self.encoder = models.vgg11(pretrained=pretrained).features
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.center = DecoderBlock(((num_filters * 8) * 2), ((num_filters * 8) * 2), (num_filters * 8))
        self.dec5 = DecoderBlock((num_filters * (16 + 8)), ((num_filters * 8) * 2), (num_filters * 8))
        self.dec4 = DecoderBlock((num_filters * (16 + 8)), ((num_filters * 8) * 2), (num_filters * 4))
        self.dec3 = DecoderBlock((num_filters * (8 + 4)), ((num_filters * 4) * 2), (num_filters * 2))
        self.dec2 = DecoderBlock((num_filters * (4 + 2)), ((num_filters * 2) * 2), num_filters)
        self.dec1 = ConvRelu((num_filters * (2 + 1)), num_filters)
        self.final = nn.Conv(num_filters, 1, 1)

    def execute(self, x):
        conv1 = nn.relu(self.conv1(x))
        conv2 = nn.relu(self.conv2(self.pool(conv1)))
        conv3s = nn.relu(self.conv3s(self.pool(conv2)))
        conv3 = nn.relu(self.conv3(conv3s))
        conv4s = nn.relu(self.conv4s(self.pool(conv3)))
        conv4 = nn.relu(self.conv4(conv4s))
        conv5s = nn.relu(self.conv5s(self.pool(conv4)))
        conv5 = nn.relu(self.conv5(conv5s))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(jt.contrib.concat([center, conv5], dim=1))
        dec4 = self.dec4(jt.contrib.concat([dec5, conv4], dim=1))
        dec3 = self.dec3(jt.contrib.concat([dec4, conv3], dim=1))
        dec2 = self.dec2(jt.contrib.concat([dec3, conv2], dim=1))
        dec1 = self.dec1(jt.contrib.concat([dec2, conv1], dim=1))
        return self.final(dec1)

class Interpolate(nn.Module):

    def __init__(self, size: int=None, scale_factor: int=None, mode: str='nearest', align_corners: bool=False):
        super().__init__()
        self.interp = nn.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def execute(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x

class DecoderBlockV2(nn.Module):

    def __init__(self, in_channels: int, middle_channels: int, out_channels: int, is_deconv: bool=True):
        super().__init__()
        self.in_channels = in_channels
        if is_deconv:
            '\n                Paramaters for Deconvolution were chosen to avoid artifacts, following\n                link https://distill.pub/2016/deconv-checkerboard/\n            '
            self.block = nn.Sequential(ConvRelu(in_channels, middle_channels), nn.ConvTranspose(middle_channels, out_channels, 4, stride=2, padding=1), nn.ReLU())
        else:
            self.block = nn.Sequential(Interpolate(scale_factor=2, mode='bilinear'), ConvRelu(in_channels, middle_channels), ConvRelu(middle_channels, out_channels))

    def execute(self, x):
        return self.block(x)

class UNet16(nn.Module):

    def __init__(self, num_classes: int=1, num_filters: int=32, pretrained: bool=False, is_deconv: bool=False):
        '\n        Args:\n            num_classes:\n            num_filters:\n            pretrained:\n                False - no pre-trained network used\n                True - encoder pre-trained with VGG16\n            is_deconv:\n                False: bilinear interpolation is used in decoder\n                True: deconvolution is used in decoder\n        '
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.Pool(2, stride=2, op='maximum')
        self.encoder = vgg16(pretrained=pretrained).features
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(self.encoder[0], self.relu, self.encoder[2], self.relu)
        self.conv2 = nn.Sequential(self.encoder[5], self.relu, self.encoder[7], self.relu)
        self.conv3 = nn.Sequential(self.encoder[10], self.relu, self.encoder[12], self.relu, self.encoder[14], self.relu)
        self.conv4 = nn.Sequential(self.encoder[17], self.relu, self.encoder[19], self.relu, self.encoder[21], self.relu)
        self.conv5 = nn.Sequential(self.encoder[24], self.relu, self.encoder[26], self.relu, self.encoder[28], self.relu)
        self.center = DecoderBlockV2(512, ((num_filters * 8) * 2), (num_filters * 8), is_deconv)
        self.dec5 = DecoderBlockV2((512 + (num_filters * 8)), ((num_filters * 8) * 2), (num_filters * 8), is_deconv)
        self.dec4 = DecoderBlockV2((512 + (num_filters * 8)), ((num_filters * 8) * 2), (num_filters * 8), is_deconv)
        self.dec3 = DecoderBlockV2((256 + (num_filters * 8)), ((num_filters * 4) * 2), (num_filters * 2), is_deconv)
        self.dec2 = DecoderBlockV2((128 + (num_filters * 2)), ((num_filters * 2) * 2), num_filters, is_deconv)
        self.dec1 = ConvRelu((64 + num_filters), num_filters)
        self.final = nn.Conv(num_filters, num_classes, 1)

    def execute(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(jt.contrib.concat([center, conv5], dim=1))
        dec4 = self.dec4(jt.contrib.concat([dec5, conv4], dim=1))
        dec3 = self.dec3(jt.contrib.concat([dec4, conv3], dim=1))
        dec2 = self.dec2(jt.contrib.concat([dec3, conv2], dim=1))
        dec1 = self.dec1(jt.contrib.concat([dec2, conv1], dim=1))
        return self.final(dec1)