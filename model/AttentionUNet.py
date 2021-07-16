import jittor as jt
from jittor import init
from jittor import nn

def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if (hasattr(m, 'weight') and ((classname.find('Conv') != (- 1)) or (classname.find('Linear') != (- 1)))):
            if (init_type == 'normal'):
                init.gauss_(m.weight.data, mean=0.0, std=gain)
            elif (init_type == 'xavier'):
                init.xavier_normal_(m.weight.data, gain=gain)
            elif (init_type == 'kaiming'):
                
                raise AttributeError("origin source: <init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')>, kaiming_normal_ in Jittor has no Attribute a")
            elif (init_type == 'orthogonal'):
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(('initialization method [%s] is not implemented' % init_type))
            if (hasattr(m, 'bias') and (m.bias is not None)):
                init.constant_(m.bias.data, value=0.0)
        elif (classname.find('BatchNorm2d') != (- 1)):
            init.gauss_(m.weight.data, mean=1.0, std=gain)
            init.constant_(m.bias.data, value=0.0)
    print(('initialize network with %s' % init_type))
    net.apply(init_func)

class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv(ch_in, ch_out, 3, stride=1, padding=1, bias=True), 
            nn.BatchNorm(ch_out), 
            nn.ReLU(), 
            nn.Conv(ch_out, ch_out, 3, stride=1, padding=1, bias=True), 
            nn.BatchNorm(ch_out), 
            nn.ReLU()
        )

    def execute(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv(ch_in, ch_out, 3, stride=1, padding=1, bias=True), 
            nn.BatchNorm(ch_out), 
            nn.ReLU()
        )

    def execute(self, x):
        x = self.up(x)
        return x

class single_conv(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv(ch_in, ch_out, 3, stride=1, padding=1, bias=True), 
            nn.BatchNorm(ch_out), 
            nn.ReLU()
        )

    def execute(self, x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(nn.Conv(F_g, F_int, 1, stride=1, padding=0, bias=True), nn.BatchNorm(F_int))
        self.W_x = nn.Sequential(nn.Conv(F_l, F_int, 1, stride=1, padding=0, bias=True), nn.BatchNorm(F_int))
        self.psi = nn.Sequential(nn.Conv(F_int, 1, 1, stride=1, padding=0, bias=True), nn.BatchNorm(1), nn.Sigmoid())
        self.relu = nn.ReLU()

    def execute(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = nn.relu((g1 + x1))
        psi = self.psi(psi)
        return (x * psi)

class AttentionUNet(nn.Module):

    def __init__(self, img_ch=3, n_classes=2):
        super(AttentionUNet, self).__init__()
        self.Maxpool = nn.Pool(2, stride=2, op='maximum')
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv(64, n_classes, 1, stride=1, padding=0)

    def execute(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = jt.contrib.concat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        x3 = self.Att4(d4, x3)
        d4 = jt.contrib.concat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3(d3, x2)
        d3 = jt.contrib.concat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att2(d2, x1)
        d2 = jt.contrib.concat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ========================================= 使用pytorch进行转换 ========================================= #
# from jittor.utils.pytorch_converter import convert

# pytorch_code="""
# import torch
# import torch.nn as nn
# from torch.nn import init


# def init_weights(net, init_type='normal', gain=0.02):
#     def init_func(m):
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find('BatchNorm2d') != -1:
#             init.normal_(m.weight.data, 1.0, gain)
#             init.constant_(m.bias.data, 0.0)

#     print('initialize network with %s' % init_type)
#     net.apply(init_func)


# class conv_block(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(conv_block,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )


#     def forward(self,x):
#         x = self.conv(x)
#         return x


# class up_conv(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(up_conv,self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
# 		    nn.BatchNorm2d(ch_out),
# 			nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.up(x)
#         return x


# class single_conv(nn.Module):
#     def __init__(self,ch_in,ch_out):
#         super(single_conv,self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self,x):
#         x = self.conv(x)
#         return x


# class Attention_block(nn.Module):
#     def __init__(self,F_g,F_l,F_int):
#         super(Attention_block,self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int),
#             )
        
#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
        
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self,g,x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1+x1)
#         psi = self.psi(psi)

#         return x*psi


# class AttU_Net(nn.Module):
#     def __init__(self,img_ch=3, n_classes=2):
#         super(AttU_Net,self).__init__()
        
#         self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

#         self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
#         self.Conv2 = conv_block(ch_in=64,ch_out=128)
#         self.Conv3 = conv_block(ch_in=128,ch_out=256)
#         self.Conv4 = conv_block(ch_in=256,ch_out=512)
#         self.Conv5 = conv_block(ch_in=512,ch_out=1024)

#         self.Up5 = up_conv(ch_in=1024,ch_out=512)
#         self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
#         self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

#         self.Up4 = up_conv(ch_in=512,ch_out=256)
#         self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
#         self.Up3 = up_conv(ch_in=256,ch_out=128)
#         self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
#         self.Up2 = up_conv(ch_in=128,ch_out=64)
#         self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

#         self.Conv_1x1 = nn.Conv2d(64,n_classes,kernel_size=1,stride=1,padding=0)


#     def forward(self,x):
#         # encoding path
#         x1 = self.Conv1(x)

#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
        
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)

#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)

#         x5 = self.Maxpool(x4)
#         x5 = self.Conv5(x5)

#         # decoding + concat path
#         d5 = self.Up5(x5)
#         x4 = self.Att5(d5,x4)
#         d5 = torch.cat((x4,d5),dim=1)        
#         d5 = self.Up_conv5(d5)
        
#         d4 = self.Up4(d5)
#         x3 = self.Att4(d4,x3)
#         d4 = torch.cat((x3,d4),dim=1)
#         d4 = self.Up_conv4(d4)

#         d3 = self.Up3(d4)
#         x2 = self.Att3(d3,x2)
#         d3 = torch.cat((x2,d3),dim=1)
#         d3 = self.Up_conv3(d3)

#         d2 = self.Up2(d3)
#         x1 = self.Att2(d2,x1)
#         d2 = torch.cat((x1,d2),dim=1)
#         d2 = self.Up_conv2(d2)

#         d1 = self.Conv_1x1(d2)

#         return d1
# # """

# jittor_code = convert(pytorch_code)
# print(jittor_code)