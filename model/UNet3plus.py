'''
pytorch implementation: https://github.com/avBuffer/UNet3plus_pth/blob/master/unet/UNet3Plus.py
'''

import jittor as jt
from jittor import init
import numpy as np
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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x):
        return self.conv(x)

class UNet3Plus(nn.Module):

    def __init__(self, in_ch=3, n_classes=2, bilinear=True):
        super(UNet3Plus, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        
        ## -------------Encoder--------------
        self.conv1 = DoubleConv(in_ch, filters[0])
        self.maxpool1 = nn.Pool(2, op='maximum')
        self.conv2 = DoubleConv(filters[0], filters[1])
        self.maxpool2 = nn.Pool(2, op='maximum')
        self.conv3 = DoubleConv(filters[1], filters[2])
        self.maxpool3 = nn.Pool(2, op='maximum')
        self.conv4 = DoubleConv(filters[2], filters[3])
        self.maxpool4 = nn.Pool(2, op='maximum')
        self.conv5 = DoubleConv(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = (self.CatChannels * self.CatBlocks)

        '''stage 4d'''
        # h1->512*512, hd4->64*64, Pooling 8 times
        self.h1_PT_hd4 = nn.Pool(8, stride=8, ceil_mode=True, op='maximum')
        self.h1_PT_hd4_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->256*256, hd4->64*64, Pooling 4 times
        self.h2_PT_hd4 = nn.Pool(4, stride=4, ceil_mode=True, op='maximum')
        self.h2_PT_hd4_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->128*128, hd4->64*64, Pooling 2 times
        self.h3_PT_hd4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h3_PT_hd4_conv = nn.Conv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->64*64, hd4->64*64, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->32*32, hd4->64*64, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm(self.UpChannels)
        self.relu4d_1 = nn.ReLU()

        '''stage 3d'''
        # h1->512*512, hd3->128*128, Pooling 4 times
        self.h1_PT_hd3 = nn.Pool(4, stride=4, ceil_mode=True, op='maximum')
        self.h1_PT_hd3_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->256*256, hd3->128*128, Pooling 2 times
        self.h2_PT_hd3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h2_PT_hd3_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->128*128, hd3->128*128, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->64*64, hd4->128*128, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->32*32, hd4->128*128, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm(self.UpChannels)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d'''
        # h1->512*512, hd2->256*256, Pooling 2 times
        self.h1_PT_hd2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h1_PT_hd2_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->256*256, hd2->256*256, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->128*128, hd2->256*256, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->64*64, hd2->256*256, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->32*32, hd2->256*256, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm(self.UpChannels)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->512*512, hd1->512*512, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->256*256, hd1->512*512, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->128*128, hd1->512*512, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd3_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->64*64, hd1->512*512, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd4_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->32*32, hd1->512*512, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm(self.UpChannels)
        self.relu1d_1 = nn.ReLU()

        # output
        self.outc = OutConv(self.UpChannels, n_classes)

    def execute(self, inputs):
        h1 = self.conv1(inputs)
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(jt.contrib.concat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), dim=1))))
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(jt.contrib.concat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), dim=1))))
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(jt.contrib.concat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), dim=1))))
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(jt.contrib.concat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), dim=1))))
        out = self.outc(hd1)
        return out

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


'''
    UNet 3+ with deep supervision
'''
class UNet3Plus_DeepSup(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, bilinear=True):
        super(UNet3Plus_DeepSup, self).__init__()


        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = DoubleConv(in_ch, filters[0])
        self.maxpool1 = nn.Pool(2, op='maximum')
        self.conv2 = DoubleConv(filters[0], filters[1])
        self.maxpool2 = nn.Pool(2, op='maximum')
        self.conv3 = DoubleConv(filters[1], filters[2])
        self.maxpool3 = nn.Pool(2, op='maximum')
        self.conv4 = DoubleConv(filters[2], filters[3])
        self.maxpool4 = nn.Pool(2, op='maximum')
        self.conv5 = DoubleConv(filters[3], filters[4])

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = (self.CatChannels * self.CatBlocks)

        '''stage 4d'''
        # h1->512*512, hd4->64*64, Pooling 8 times
        self.h1_PT_hd4 = nn.Pool(8, stride=8, ceil_mode=True, op='maximum')
        self.h1_PT_hd4_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU()

        # h2->256*256, hd4->64*64, Pooling 4 times
        self.h2_PT_hd4 = nn.Pool(4, stride=4, ceil_mode=True, op='maximum')
        self.h2_PT_hd4_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU()

        # h3->128*128, hd4->64*64, Pooling 2 times
        self.h3_PT_hd4 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h3_PT_hd4_conv = nn.Conv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU()

        # h4->64*64, hd4->64*64, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU()

        # hd5->32*32, hd4->64*64, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU()

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm(self.UpChannels)
        self.relu4d_1 = nn.ReLU()


        '''stage 3d'''
        # h1->512*512, hd3->128*128, Pooling 4 times
        self.h1_PT_hd3 = nn.Pool(4, stride=4, ceil_mode=True, op='maximum')
        self.h1_PT_hd3_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU()

        # h2->256*256, hd3->128*128, Pooling 2 times
        self.h2_PT_hd3 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h2_PT_hd3_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU()

        # h3->128*128, hd3->128*128, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU()

        # hd4->64*64, hd4->128*128, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU()

        # hd5->32*32, hd4->128*128, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU()

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm(self.UpChannels)
        self.relu3d_1 = nn.ReLU()

        '''stage 2d'''
        # h1->512*512, hd2->256*256, Pooling 2 times
        self.h1_PT_hd2 = nn.Pool(2, stride=2, ceil_mode=True, op='maximum')
        self.h1_PT_hd2_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU()

        # h2->256*256, hd2->256*256, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU()

        # hd3->128*128, hd2->256*256, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU()

        # hd4->64*64, hd2->256*256, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU()

        # hd5->32*32, hd2->256*256, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU()

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm(self.UpChannels)
        self.relu2d_1 = nn.ReLU()

        '''stage 1d'''
        # h1->512*512, hd1->512*512, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU()

        # hd2->256*256, hd1->512*512, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd2_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU()

        # hd3->128*128, hd1->512*512, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd3_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU()

        # hd4->64*64, hd1->512*512, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd4_UT_hd1_conv = nn.Conv(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU()

        # hd5->32*32, hd1->512*512, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.hd5_UT_hd1_conv = nn.Conv(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU()

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm(self.UpChannels)
        self.relu1d_1 = nn.ReLU()

        # -------------Bilinear Upsampling--------------
        self.upscore6 = nn.Upsample(scale_factor=32,mode='bilinear')
        self.upscore5 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], n_classes, 3, padding=1)

    def execute(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)
        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)
        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)
        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(jt.contrib.concat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), dim=1))))
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(jt.contrib.concat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), dim=1))))
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(jt.contrib.concat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), dim=1))))
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(jt.contrib.concat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), dim=1))))

        d5 = self.outconv5(hd5)
        d5 = self.upscore5(d5)      # 32->512

        d4 = self.outconv4(hd4)
        d4 = self.upscore4(d4)      # 64->512

        d3 = self.outconv3(hd3)
        d3 = self.upscore3(d3)      # 128->512

        d2 = self.outconv2(hd2)
        d2 = self.upscore2(d2)      # 256->512

        d1 = self.outconv1(hd1)     # 512->512

        return jt.sigmoid(d1), jt.sigmoid(d2), jt.sigmoid(d3), jt.sigmoid(d4), jt.sigmoid(d5)

    def get_loss(self, target, d1, d2, d3, d4, d5, ignore_index=None):

        loss1 = nn.cross_entropy_loss(d1, target, ignore_index=ignore_index)        # tar loss
        loss2 = nn.cross_entropy_loss(d2, target, ignore_index=ignore_index)
        loss3 = nn.cross_entropy_loss(d3, target, ignore_index=ignore_index)
        loss4 = nn.cross_entropy_loss(d4, target, ignore_index=ignore_index)
        loss5 = nn.cross_entropy_loss(d5, target, ignore_index=ignore_index)

        loss = loss1 + loss2 + loss3 + loss4 + loss5                                # backward
        print("l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item()))

        return loss1, loss

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    model = UNet3Plus_DeepSup()
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y[0].shape)
    # _ = y.data

if __name__ == '__main__':
    main()

# from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import numpy as np
# import torch
# import torch.nn as nn

# class unetConv2(nn.Module):
#     def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
#         super(unetConv2, self).__init__()
#         self.n = n
#         self.ks = ks
#         self.stride = stride
#         self.padding = padding
#         s = stride
#         p = padding
        
#         if is_batchnorm:
#             for i in range(1, n + 1):
#                 conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
#                                      nn.BatchNorm2d(out_size), nn.ReLU(inplace=True),)
#                 setattr(self, 'conv%d' % i, conv)
#                 in_size = out_size
#         else:
#             for i in range(1, n + 1):
#                 conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.ReLU(inplace=True), )
#                 setattr(self, 'conv%d' % i, conv)
#                 in_size = out_size


#     def forward(self, inputs):
#         x = inputs
#         for i in range(1, self.n + 1):
#             conv = getattr(self, 'conv%d' % i)
#             x = conv(x)
#         return x

# class UNet3Plus(nn.Module):
#     def __init__(self, n_channels=3, n_classes=1, bilinear=True, feature_scale=4,
#                  is_deconv=True, is_batchnorm=True):
#         super(UNet3Plus, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.feature_scale = feature_scale
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#         filters = [64, 128, 256, 512, 1024]

#         ## -------------Encoder--------------
#         self.conv1 = unetConv2(self.n_channels, filters[0], self.is_batchnorm)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)

#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)

#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2)

#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2)

#         self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)

#         ## -------------Decoder--------------
#         self.CatChannels = filters[0]
#         self.CatBlocks = 5
#         self.UpChannels = self.CatChannels * self.CatBlocks

#         '''stage 4d'''
#         # h1->320*320, hd4->40*40, Pooling 8 times
#         self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
#         self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
#         self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

#         # h2->160*160, hd4->40*40, Pooling 4 times
#         self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
#         self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
#         self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

#         # h3->80*80, hd4->40*40, Pooling 2 times
#         self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
#         self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

#         # h4->40*40, hd4->40*40, Concatenation
#         self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
#         self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

#         # hd5->20*20, hd4->40*40, Upsample 2 times
#         self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
#         self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

#         # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
#         self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
#         self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
#         self.relu4d_1 = nn.ReLU(inplace=True)

#         '''stage 3d'''
#         # h1->320*320, hd3->80*80, Pooling 4 times
#         self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
#         self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
#         self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

#         # h2->160*160, hd3->80*80, Pooling 2 times
#         self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
#         self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

#         # h3->80*80, hd3->80*80, Concatenation
#         self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
#         self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

#         # hd4->40*40, hd4->80*80, Upsample 2 times
#         self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

#         # hd5->20*20, hd4->80*80, Upsample 4 times
#         self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
#         self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

#         # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
#         self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
#         self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
#         self.relu3d_1 = nn.ReLU(inplace=True)

#         '''stage 2d '''
#         # h1->320*320, hd2->160*160, Pooling 2 times
#         self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
#         self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
#         self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

#         # h2->160*160, hd2->160*160, Concatenation
#         self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
#         self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

#         # hd3->80*80, hd2->160*160, Upsample 2 times
#         self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

#         # hd4->40*40, hd2->160*160, Upsample 4 times
#         self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

#         # hd5->20*20, hd2->160*160, Upsample 8 times
#         self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
#         self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
#         self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

#         # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
#         self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
#         self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
#         self.relu2d_1 = nn.ReLU(inplace=True)

#         '''stage 1d'''
#         # h1->320*320, hd1->320*320, Concatenation
#         self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
#         self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

#         # hd2->160*160, hd1->320*320, Upsample 2 times
#         self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
#         self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

#         # hd3->80*80, hd1->320*320, Upsample 4 times
#         self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
#         self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

#         # hd4->40*40, hd1->320*320, Upsample 8 times
#         self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
#         self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
#         self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

#         # hd5->20*20, hd1->320*320, Upsample 16 times
#         self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
#         self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
#         self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
#         self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

#         # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
#         self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
#         self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
#         self.relu1d_1 = nn.ReLU(inplace=True)

#         # output
#         self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)


#     def forward(self, inputs):
#         ## -------------Encoder-------------
#         h1 = self.conv1(inputs)  # h1->320*320*64

#         h2 = self.maxpool1(h1)
#         h2 = self.conv2(h2)  # h2->160*160*128

#         h3 = self.maxpool2(h2)
#         h3 = self.conv3(h3)  # h3->80*80*256

#         h4 = self.maxpool3(h3)
#         h4 = self.conv4(h4)  # h4->40*40*512

#         h5 = self.maxpool4(h4)
#         hd5 = self.conv5(h5)  # h5->20*20*1024

#         ## -------------Decoder-------------
#         h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
#         h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
#         h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
#         h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
#         hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
#         hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1)))) # hd4->40*40*UpChannels

#         h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
#         h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
#         h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
#         hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
#         hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
#         hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1)))) # hd3->80*80*UpChannels

#         h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
#         h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
#         hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
#         hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
#         hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
#         hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1)))) # hd2->160*160*UpChannels

#         h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
#         hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
#         hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
#         hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
#         hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
#         hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1)))) # hd1->320*320*UpChannels

#         d1 = self.outconv1(hd1)  # d1->320*320*n_classes
#         return d1
# """

# jittor_code = convert(pytorch_code)
# print(jittor_code)