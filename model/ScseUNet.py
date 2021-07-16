"""
SCSE + U-Net
pytorch implementation: https://blog.csdn.net/qq_39071739/article/details/107111337
"""

import jittor as jt
from jittor import init
from jittor import nn

class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)
        self.channel_gate = ChannelGate2d(in_ch)

    def execute(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = (g1 + g2)
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()
        self.linear_1 = nn.Linear(in_ch, (in_ch // r))
        self.linear_2 = nn.Linear((in_ch // r), in_ch)

    def execute(self, x):
        input_x = x
        x = x.view((*x.shape[:(- 2)], (- 1))).mean((- 1))
        x = self.linear_1(x)
        x = nn.relu(x)
        x = self.linear_2(x)
        x = x.unsqueeze((- 1)).unsqueeze((- 1))
        # x = nn.sigmoid(x)
        x = nn.softmax(x)
        x = (input_x * x)
        return x

class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()
        self.conv = nn.Conv(in_ch, 1, 1, stride=1)

    def execute(self, x):
        input_x = x
        x = self.conv(x)
        # x = nn.sigmoid(x)
        x = nn.softmax(x)
        x = (input_x * x)
        return x

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
        self.spa_cha_gate = SCSE(out_channels)

    def execute(self, x1, x2 = None):
        x = self.up(x1)
        x = jt.contrib.concat([x2, x], dim=1)
        x = self.conv(x)
        x = self.spa_cha_gate(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)

class SCSEUnet(nn.Module):
    def __init__(self, in_ch = 3, n_classes = 2, bilinear = True):
        super(SCSEUnet, self).__init__()
        self.conv_encode1 = nn.Sequential(DoubleConv(in_channels=in_ch, out_channels=64), SCSE(64))
        self.conv_pool1 = nn.Pool(2, stride=2, op='maximum')
        self.conv_encode2 = nn.Sequential(DoubleConv(in_channels=64, out_channels=128), SCSE(128))
        self.conv_pool2 = nn.Pool(2, stride=2, op='maximum')
        self.conv_encode3 = nn.Sequential(DoubleConv(in_channels=128, out_channels=256), SCSE(256))
        self.conv_pool3 = nn.Pool(2, stride=2, op='maximum')
        self.conv_encode4 = nn.Sequential(DoubleConv(in_channels=256, out_channels=512), SCSE(512))
        self.conv_pool4 = nn.Pool(2, stride=2, op='maximum')
        factor = (2 if bilinear else 1)
        self.bottleneck = nn.Sequential(DoubleConv(in_channels=512, out_channels=(1024 // factor)), SCSE((1024 // factor)))
        self.conv_decode4 = Up(1024, (512 // factor), bilinear)
        self.conv_decode3 = Up(512, (256 // factor), bilinear)
        self.conv_decode2 = Up(256, (128 // factor), bilinear)
        self.conv_decode1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def execute(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)
        bottleneck = self.bottleneck(encode_pool4)
        decode_block4 = self.conv_decode4(bottleneck, encode_block4)
        decode_block3 = self.conv_decode3(decode_block4, encode_block3)
        decode_block2 = self.conv_decode2(decode_block3, encode_block2)
        decode_block1 = self.conv_decode1(decode_block2, encode_block1)
        out = self.outc(decode_block1)
        return out

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    model = SCSEUnet()
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()

# from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torchsummary import summary


# # SCSE模块
# class SCSE(nn.Module):
#     def __init__(self, in_ch):
#         super(SCSE, self).__init__()
#         self.spatial_gate = SpatialGate2d(in_ch, 16)  # 16
#         self.channel_gate = ChannelGate2d(in_ch)

#     def forward(self, x):
#         g1 = self.spatial_gate(x)
#         g2 = self.channel_gate(x)
#         x = g1 + g2  # x = g1*x + g2*x
#         return x


# # 空间门控
# class SpatialGate2d(nn.Module):
#     def __init__(self, in_ch, r=16):
#         super(SpatialGate2d, self).__init__()

#         self.linear_1 = nn.Linear(in_ch, in_ch // r)
#         self.linear_2 = nn.Linear(in_ch // r, in_ch)

#     def forward(self, x):
#         input_x = x

#         x = x.view(*(x.shape[:-2]), -1).mean(-1)
#         x = nn.relu(self.linear_1(x), inplace=True)
#         x = self.linear_2(x)
#         x = x.unsqueeze(-1).unsqueeze(-1)
#         x = torch.sigmoid(x)
#         x = input_x * x

#         return x


# # 通道门控
# class ChannelGate2d(nn.Module):
#     def __init__(self, in_ch):
#         super(ChannelGate2d, self).__init__()

#         self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

#     def forward(self, x):
#         input_x = x
#         x = self.conv(x)
#         x = torch.sigmoid(x)
#         x = input_x * x

#         return x


# # 编码连续卷积层
# def contracting_block(in_channels, out_channels):
#     block = torch.nn.Sequential(
#         nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(),
#         nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU()
#     )
#     return block


# # 解码上采样卷积层
# class expansive_block(nn.Module):
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(expansive_block, self).__init__()
#         self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(3, 3), stride=2, padding=1,
#                                      output_padding=1, dilation=1)
#         self.block = nn.Sequential(
#             nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=mid_channels, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(),
#             nn.Conv2d(kernel_size=(3, 3), in_channels=mid_channels, out_channels=out_channels, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU()
#         )
#         self.spa_cha_gate = SCSE(out_channels)

#     def forward(self, d, e=None):
#         d = self.up(d)
#         # d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=True)
#         # concat
#         if e is not None:
#             cat = torch.cat([e, d], dim=1)
#             out = self.block(cat)
#         else:
#             out = self.block(d)
#         out = self.spa_cha_gate(out)
#         return out


# # 输出层
# def final_block(in_channels, out_channels):
#     block = nn.Sequential(
#         nn.Conv2d(kernel_size=(1, 1), in_channels=in_channels, out_channels=out_channels),
#         # nn.BatchNorm2d(out_channels),
#         # nn.ReLU()
#     )
#     return block


# # SCSE U-Net
# class SCSEUnet(nn.Module):

#     def __init__(self, in_channel, out_channel):
#         super(SCSEUnet, self).__init__()
#         # Encode
#         self.conv_encode1 = nn.Sequential(contracting_block(in_channels=in_channel, out_channels=32), SCSE(32))
#         self.conv_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv_encode2 = nn.Sequential(contracting_block(in_channels=32, out_channels=64), SCSE(64))
#         self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv_encode3 = nn.Sequential(contracting_block(in_channels=64, out_channels=128), SCSE(128))
#         self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv_encode4 = nn.Sequential(contracting_block(in_channels=128, out_channels=256), SCSE(256))
#         self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         # Bottleneck
#         self.bottleneck = torch.nn.Sequential(
#             nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             SCSE(512)
#         )
        
#         # Decode
#         self.conv_decode4 = expansive_block(512, 256, 256)
#         self.conv_decode3 = expansive_block(256, 128, 128)
#         self.conv_decode2 = expansive_block(128, 64, 64)
#         self.conv_decode1 = expansive_block(64, 32, 32)
#         self.final_layer = final_block(32, out_channel)

#     def forward(self, x):
#         # set_trace()
#         # Encode
#         encode_block1 = self.conv_encode1(x)
#         encode_pool1 = self.conv_pool1(encode_block1)
#         encode_block2 = self.conv_encode2(encode_pool1)
#         encode_pool2 = self.conv_pool2(encode_block2)
#         encode_block3 = self.conv_encode3(encode_pool2)
#         encode_pool3 = self.conv_pool3(encode_block3)
#         encode_block4 = self.conv_encode4(encode_pool3)
#         encode_pool4 = self.conv_pool4(encode_block4)

#         # Bottleneck
#         bottleneck = self.bottleneck(encode_pool4)

#         # Decode
#         decode_block4 = self.conv_decode4(bottleneck, encode_block4)
#         decode_block3 = self.conv_decode3(decode_block4, encode_block3)
#         decode_block2 = self.conv_decode2(decode_block3, encode_block2)
#         decode_block1 = self.conv_decode1(decode_block2, encode_block1)

#         final_layer = self.final_layer(decode_block1)
#         out = torch.sigmoid(final_layer)  # 可注释，根据情况

#         return out
# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)