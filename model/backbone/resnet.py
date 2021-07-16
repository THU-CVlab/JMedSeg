import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat


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
        self.inplanes = 128
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
        self.conv1 = nn.Sequential(
            nn.Conv(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.Conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False))


        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1)

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

        #x = self.bn1(x)
        #x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)


        x_0 = x
        x_1 = self.layer2(x)
        x_2 = self.layer3(x_1)
        x_3 = self.layer4(x_2)
        return x_0, x_1, x_2, x_3

def load_pretrained_model(model, params_path):
    pretrained_dict = torch.load(params_path)
    model_dict = {}
    param_name = model.parameters()
    name_list = [item.name() for item in param_name]
    for k, v in pretrained_dict.items():
        if k in name_list:
            # print (k)
            model_dict[k] = v
    model.load_parameters(model_dict)

def resnet50(output_stride):
    model = ResNet(Bottleneck, [3,4,6,3], output_stride)
    # load imagenet pretrain model
    #model_path = './pretrained_models/resnet50.pth'
    model_path = 'model/pretrained_models/resnet50-ebb6acbb.pth'
    load_pretrained_model(model, model_path)
    return model

def resnet101(output_stride):
    model = ResNet(Bottleneck, [3,4,23,3], output_stride)
    # load imagenet pretrain model
    #model_path = './pretrained_models/resnet101.pth'
    model_path = 'model/pretrained_models/resnet101-2a57e44d.pth'
    # load_pretrained_model(model, model_path)
    return model