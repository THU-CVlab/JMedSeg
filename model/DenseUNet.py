'''
pytorch implementation: https://github.com/Merofine/DenseUnet_BraTs/blob/master/denseUnet.py
'''

import jittor as jt
from jittor import init
from jittor import nn

class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv(filters, filters, 3, padding=1))
            self.bn_list.append(nn.BatchNorm(filters))

    def execute(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if (i > 0):
                for j in range(i):
                    temp_out += outs[j]
            outs.append(nn.relu(self.bn_list[i](temp_out)))
        out_final = outs[(- 1)]
        del outs
        return out_final

class Down_sample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.Pool(kernel_size, stride=stride, op='maximum')

    def execute(self, x):
        y = self.down_sample_layer(x)
        return (y, x)

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
    '''
    in_channels: 是x的channel，而不是concat之后的channel
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = DoubleConv(in_channels * 2, out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose(in_channels, in_channels, 2, stride=2)
            self.conv = DoubleConv(in_channels * 2, out_channels, out_channels)

    def execute(self, x1, x2):
        x = self.up(x1)
        x = jt.contrib.concat([x2, x], dim=1)
        x = self.conv(x)
        return x

class DenseUNet(nn.Module):

    def __init__(self, in_ch = 3, n_classes = 2, bilinear=True):
        num_conv = 4
        filters = 64
        super(DenseUNet, self).__init__()
        self.conv1 = nn.Conv(in_ch, filters, 1)
        self.d1 = Single_level_densenet(filters, num_conv)
        self.down1 = Down_sample()
        self.d2 = Single_level_densenet(filters, num_conv)
        self.down2 = Down_sample()
        self.d3 = Single_level_densenet(filters, num_conv)
        self.down3 = Down_sample()
        self.d4 = Single_level_densenet(filters, num_conv)
        self.down4 = Down_sample()
        self.bottom = Single_level_densenet(filters, num_conv)
        self.up4 = Up(filters, filters, bilinear)
        self.u4 = Single_level_densenet(filters, num_conv)
        self.up3 = Up(filters, filters, bilinear)
        self.u3 = Single_level_densenet(filters, num_conv)
        self.up2 = Up(filters, filters, bilinear)
        self.u2 = Single_level_densenet(filters, num_conv)
        self.up1 = Up(filters, filters, bilinear)
        self.u1 = Single_level_densenet(filters, num_conv)
        self.outconv = nn.Conv(filters, n_classes, 1)

    def execute(self, x):
        x = self.conv1(x)
        (x, y1) = self.down1(self.d1(x))
        (x, y2) = self.down1(self.d2(x))
        (x, y3) = self.down1(self.d3(x))
        (x, y4) = self.down1(self.d4(x))
        # print(y1.shape, y2.shape, y3.shape, y4.shape)   # [-1,64,512,512,] [-1,64,256,256,] [-1,64,128,128,] [-1,64,64,64,]
        x = self.bottom(x)              # [-1, 64, 32, 32]
        x = self.u4(self.up4(x, y4))    # [-1, 64, 64, 64,]
        x = self.u3(self.up3(x, y3))    # [-1, 64, 128, 128,]
        x = self.u2(self.up2(x, y2))    # [-1, 64, 256, 256,]
        x = self.u1(self.up1(x, y1))    # [-1, 64, 512, 512,]
        x1 = self.outconv(x)            # [-1, 2, 512, 512,]
        return x1

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    model = DenseUnet()
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()

# from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import torch
# import torch.nn as nn


# class Single_level_densenet(nn.Module):
#     def __init__(self, filters, num_conv=4):
#         super(Single_level_densenet, self).__init__()
#         self.num_conv = num_conv
#         self.conv_list = nn.ModuleList()
#         self.bn_list = nn.ModuleList()
#         for i in range(self.num_conv):
#             self.conv_list.append(nn.Conv2d(filters, filters, 3, padding=1))
#             self.bn_list.append(nn.BatchNorm2d(filters))

#     def forward(self, x):
#         outs = []
#         outs.append(x)
#         for i in range(self.num_conv):
#             temp_out = self.conv_list[i](outs[i])
#             if i > 0:
#                 for j in range(i):
#                     temp_out += outs[j]
#             outs.append(nn.relu(self.bn_list[i](temp_out)))
#         out_final = outs[-1]
#         del outs
#         return out_final


# class Down_sample(nn.Module):
#     def __init__(self, kernel_size=2, stride=2):
#         super(Down_sample, self).__init__()
#         self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)

#     def forward(self, x):
#         y = self.down_sample_layer(x)
#         return y, x


# class Upsample_n_Concat(nn.Module):
#     def __init__(self, filters):
#         super(Upsample_n_Concat, self).__init__()
#         self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding=1, stride=2)
#         self.conv = nn.Conv2d(2 * filters, filters, 3, padding=1)
#         self.bn = nn.BatchNorm2d(filters)

#     def forward(self, x, y):
#         x = self.upsample_layer(x)
#         x = torch.cat([x, y], dim=1)
#         x = nn.relu(self.bn(self.conv(x)))
#         return x


# class DenseUnet(nn.Module):
#     def __init__(self, args):
#         self.args = args
#         num_conv = 4
#         filters = 64
#         in_chan = 4
#         out_chan = 3
#         super(DenseUnet, self).__init__()
#         self.conv1 = nn.Conv2d(in_chan, filters, 1)
#         self.d1 = Single_level_densenet(filters, num_conv)
#         self.down1 = Down_sample()
#         self.d2 = Single_level_densenet(filters, num_conv)
#         self.down2 = Down_sample()
#         self.d3 = Single_level_densenet(filters, num_conv)
#         self.down3 = Down_sample()
#         self.d4 = Single_level_densenet(filters, num_conv)
#         self.down4 = Down_sample()
#         self.bottom = Single_level_densenet(filters, num_conv)
#         self.up4 = Upsample_n_Concat(filters)
#         self.u4 = Single_level_densenet(filters, num_conv)
#         self.up3 = Upsample_n_Concat(filters)
#         self.u3 = Single_level_densenet(filters, num_conv)
#         self.up2 = Upsample_n_Concat(filters)
#         self.u2 = Single_level_densenet(filters, num_conv)
#         self.up1 = Upsample_n_Concat(filters)
#         self.u1 = Single_level_densenet(filters, num_conv)
#         self.outconv = nn.Conv2d(filters, out_chan, 1)

#     def forward(self, x):

#         x = self.conv1(x)
#         x, y1 = self.down1(self.d1(x))
#         x, y2 = self.down1(self.d2(x))
#         x, y3 = self.down1(self.d3(x))
#         x, y4 = self.down1(self.d4(x))
#         x = self.bottom(x)
#         x = self.u4(self.up4(x, y4))
#         x = self.u3(self.up3(x, y3))
#         x = self.u2(self.up2(x, y2))
#         x = self.u1(self.up1(x, y1))
#         x1 = self.outconv(x)


#         return x1
# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)