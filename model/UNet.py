# import jittor as jt
# from jittor import init
# from jittor import nn

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv(in_channels, out_channels, 3, padding=1), 
#         nn.ReLU(), 
#         nn.Conv(out_channels, out_channels, 3, padding=1), 
#         nn.ReLU()
#     )

# class UNet(nn.Module):

#     def __init__(self, n_channels, n_classes):
#         super().__init__()
#         self.dconv_down1 = double_conv(n_channels, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)
#         self.maxpool = nn.Pool(2, op='maximum')
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
#         self.dconv_up3 = double_conv((256 + 512), 256)
#         self.dconv_up2 = double_conv((128 + 256), 128)
#         self.dconv_up1 = double_conv((128 + 64), 64)
#         self.conv_last = nn.Conv(64, n_classes, 1)

#     def execute(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)
#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)
#         x = self.dconv_down4(x)
#         x = self.upsample(x)
#         x = jt.contrib.concat([x, conv3], dim=1)
#         x = self.dconv_up3(x)
#         x = self.upsample(x)
#         x = jt.contrib.concat([x, conv2], dim=1)
#         x = self.dconv_up2(x)
#         x = self.upsample(x)
#         x = jt.contrib.concat([x, conv1], dim=1)
#         x = self.dconv_up1(x)
#         out = self.conv_last(x)
#         return out

import jittor as jt
from jittor import init
from jittor import nn

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

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Pool(kernel_size=2, stride=2, op='maximum'), 
            DoubleConv(in_channels, out_channels)
        )

    def execute(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, (in_channels // 2))
        else:
            self.up = nn.ConvTranspose(in_channels, (in_channels // 2), 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        x = jt.contrib.concat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = (2 if bilinear else 1)
        self.down4 = Down(512, (1024 // factor))
        self.up1 = Up(1024, (512 // factor), bilinear)
        self.up2 = Up(512, (256 // factor), bilinear)
        self.up3 = Up(256, (128 // factor), bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def main():
    model = UNet()
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
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
    summary(model, input_size=(3, 512, 512))

if __name__ == '__main__':
    main()


# ========================================= 使用pytorch进行转换 ========================================= #

#  from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     # (convolution => [BN] => ReLU) * 2 #
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class Down(nn.Module):
#     # Downscaling with maxpool then double conv #
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     # Upscaling then double conv #
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()

#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear')
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         # Note that the parameters are different for binlinear upsampling layer and 
#         # non-binlinear upsampling layer, and deconvolution with more channels to 
#         # restore information
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         # print('x5',x5.shape)
#         x = self.up1(x5, x4)
#         # print('x',x.shape)
#         x = self.up2(x, x3)
#         # print('x',x.shape)
#         x = self.up3(x, x2)
#         # print('x',x.shape)
#         x = self.up4(x, x1)
#         # print('x',x.shape)
#         logits = self.outc(x)
#         return logits
# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)

# from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import torch
# import torch.nn as nn

# def double_conv(in_channels, out_channels):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, 3, padding=1),
#         nn.ReLU(inplace=True)
#     )   


# class UNet(nn.Module):

#     def __init__(self, n_class):
#         super().__init__()
                
#         self.dconv_down1 = double_conv(3, 64)
#         self.dconv_down2 = double_conv(64, 128)
#         self.dconv_down3 = double_conv(128, 256)
#         self.dconv_down4 = double_conv(256, 512)        

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')        
        
#         self.dconv_up3 = double_conv(256 + 512, 256)
#         self.dconv_up2 = double_conv(128 + 256, 128)
#         self.dconv_up1 = double_conv(128 + 64, 64)
        
#         self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool(conv1)

#         conv2 = self.dconv_down2(x)
#         x = self.maxpool(conv2)
        
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool(conv3)   
        
#         x = self.dconv_down4(x)
        
#         x = self.upsample(x)        
#         x = torch.cat([x, conv3], dim=1)
        
#         x = self.dconv_up3(x)
#         x = self.upsample(x)        
#         x = torch.cat([x, conv2], dim=1)       

#         x = self.dconv_up2(x)
#         x = self.upsample(x)        
#         x = torch.cat([x, conv1], dim=1)   
        
#         x = self.dconv_up1(x)
        
#         out = self.conv_last(x)
        
#         return out
# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)