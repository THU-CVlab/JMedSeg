"""
ResNet34 + U-Net
pytorch implementation: https://blog.csdn.net/qq_39071739/article/details/106873657
"""
import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101


Backbone_List = ['resnet50', 'resnet101']

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if (not mid_channels):
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv(in_channels, mid_channels, 3, padding=1), 
            nn.BatchNorm(mid_channels), 
            nn.ReLU(), 
            nn.Conv(mid_channels, out_channels, 3, padding=1), 
            nn.BatchNorm(out_channels), 
            nn.ReLU()
        )

    def execute(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, (in_channels // 2))
        else:
            self.up = nn.ConvTranspose(in_channels, (in_channels // 2), 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def execute(self, x1, x2 = None):
        x1 = jt.contrib.concat([x2, x1], dim=1)
        x = self.up(x1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)

class ResUNet(nn.Module):

    def __init__(self, in_ch = 3, n_classes=2, bilinear = True, output_stride=16, backbone = 'resnet101'):
        super(ResUNet, self).__init__()

        if not backbone in Backbone_List:
            print('Invalid Backbone! Initialized to resnet101')
            backbone = 'resnet101'
        if backbone == 'resnet50':
            self.backbone = resnet50(output_stride=output_stride)
        else:
            self.backbone = resnet101(output_stride=output_stride)

        self.backbone_name = backbone

        self.layer0 = self.backbone.conv1
        self.pooling0 = self.backbone.maxpool

        # Encode
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3

        # Bottleneck
        factor = (2 if bilinear else 1)
        self.bottleneck = DoubleConv(in_channels=1024, out_channels=(2048 // factor))
        
        # Decode
        self.conv_decode3 = Up(2048, (1024 // factor), bilinear)
        self.conv_decode2 = Up(1024, (512 // factor), bilinear)
        self.conv_decode1 = Up(512, (256 // factor), bilinear)
        self.conv_decode0 = Up(256, (128 // factor), bilinear)
        self.outc = OutConv(64, n_classes)

    def execute(self, x):
        # Encode   注意都是pooling之后的结果
        encode_block0 = self.layer0(x)                  # [-1,128,256,256]
        encode_block1 = self.pooling0(encode_block0)
        encode_block1 = self.layer1(encode_block1)      # [-1,256,128,128,]
        encode_block2 = self.layer2(encode_block1)      # [-1,512,64,64,]
        encode_block3 = self.layer3(encode_block2)      # [-1,1024,32,32,]
        

        # Bottleneck
        bottleneck = self.bottleneck(encode_block3)     # [-1,1024,32,32,]

        # Decode
        decode_block3 = self.conv_decode3(bottleneck, encode_block3)        # [-1,512,64,64,]
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)     # [-1,256,128,128,]
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)     # [-1,128,256,256,]
        decode_block0 = self.conv_decode0(decode_block1, encode_block0)     # [-1,64,512,512,]
        out = self.outc(decode_block0)                                      # [-1,2,512,512,]
        return out

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    model = ResUNet(backbone = 'resnet101')
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()

# import torch
# from torch import nn
# import torchvision.models as models
# import torch.nn.functional as F
# from torchsummary import summary

# class expansive_block(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(expansive_block, self).__init__()

#         self.block = nn.Sequential(
#             nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(mid_channels),
#             nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(out_channels)
#         )

#     def forward(self, d, e=None):
#         d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
#         # concat

#         if e is not None:
#             cat = torch.cat([e, d], dim=1)
#             out = self.block(cat)
#         else:
#             out = self.block(d)
#         return out


# def final_block(in_channels, out_channels):
#     block = nn.Sequential(
#         nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
#         nn.ReLU(),
#         nn.BatchNorm2d(out_channels),
#     )
#     return block


# class Resnet34_Unet(nn.Module):

#     def __init__(self, in_channel, out_channel, pretrained=False):
#         super(Resnet34_Unet, self).__init__()

#         self.resnet = models.resnet34(pretrained=pretrained)
#         self.layer0 = nn.Sequential(
#             self.resnet.conv1,
#             self.resnet.bn1,
#             self.resnet.relu,
#             self.resnet.maxpool
#         )

#         # Encode
#         self.layer1 = self.resnet.layer1
#         self.layer2 = self.resnet.layer2
#         self.layer3 = self.resnet.layer3
#         self.layer4 = self.resnet.layer4

#         # Bottleneck
#         self.bottleneck = torch.nn.Sequential(
#             nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=1024, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(1024),
#             nn.Conv2d(kernel_size=(3, 3), in_channels=1024, out_channels=1024, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(1024),
#             nn.MaxPool2d(kernel_size=(2, 2), stride=2)
#         )

#         # Decode
#         self.conv_decode4 = expansive_block(1024+512, 512, 512)
#         self.conv_decode3 = expansive_block(512+256, 256, 256)
#         self.conv_decode2 = expansive_block(256+128, 128, 128)
#         self.conv_decode1 = expansive_block(128+64, 64, 64)
#         self.conv_decode0 = expansive_block(64, 32, 32)
#         self.final_layer = final_block(32, out_channel)

#     def forward(self, x):
#         x = self.layer0(x)
#         # Encode
#         encode_block1 = self.layer1(x)
#         encode_block2 = self.layer2(encode_block1)
#         encode_block3 = self.layer3(encode_block2)
#         encode_block4 = self.layer4(encode_block3)

#         # Bottleneck
#         bottleneck = self.bottleneck(encode_block4)

#         # Decode
#         decode_block4 = self.conv_decode4(bottleneck, encode_block4)
#         decode_block3 = self.conv_decode3(decode_block4, encode_block3)
#         decode_block2 = self.conv_decode2(decode_block3, encode_block2)
#         decode_block1 = self.conv_decode1(decode_block2, encode_block1)
#         decode_block0 = self.conv_decode0(decode_block1)

#         final_layer = self.final_layer(decode_block0)

#         return final_layer


# flag = 0

# if flag:
#     image = torch.rand(1, 3, 572, 572)
#     Resnet34_Unet = Resnet34_Unet(in_channel=3, out_channel=1)
#     mask = Resnet34_Unet(image)
#     print(mask.shape)


