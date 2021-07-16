import time
import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat, argmax_pool

# https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-3-17-09-55-segmentation/
# 上图为DeepLabV3+论文给出的网络架构图。本教程采用ResNet为backbone。输入图像尺寸为513*513。

class Bottleneck(Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(Module):
    def __init__(self, block, layers, output_stride):
        super(ResNet, self).__init__()
        self.inplanes = 64
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        # self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation))

        return nn.Sequential(*layers)

    def execute(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = argmax_pool(x, 2, 2)
        x = self.layer1(x)

        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        return x, low_level_feat

def resnet50(output_stride):
    model = ResNet(Bottleneck, [3,4,6,3], output_stride)
    return model

def resnet101(output_stride):
    model = ResNet(Bottleneck, [3,4,23,3], output_stride)
    return model


class Single_ASPPModule(Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(Single_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm(planes)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ASPP(Module):
    def __init__(self, output_stride):
        super(ASPP, self).__init__()
        inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = Single_ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = Single_ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])
        self.global_avg_pool = nn.Sequential(GlobalPooling(),
                                             nn.Conv(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv(1280, 256, 1, bias=False)
        
        self.bn1 = nn.BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def execute(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = x5.broadcast((1,1,x4.shape[2],x4.shape[3]))
        x = concat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class GlobalPooling (Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
    def execute (self, x):
        return jt.mean(x, dims=[2,3], keepdims=1)

class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        low_level_inplanes = 256

        self.conv1 = nn.Conv(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv(256, num_classes, kernel_size=1, stride=1, bias=True))

    def execute(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x_inter = nn.resize(x, size=(low_level_feat.shape[2], low_level_feat.shape[3]) , mode='bilinear')
        x_concat = concat((x_inter, low_level_feat), dim=1)
        x = self.last_conv(x_concat)
        return x

class DeepLabV3Plus(Module):
    def __init__(self, output_stride=16, n_classes=2):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = resnet101(output_stride=output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(n_classes)

    def execute(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = nn.resize(x, size=(input.shape[2], input.shape[3]), mode='bilinear')
        return x
