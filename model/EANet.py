import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat, argmax_pool


class NormLayer(Module):
    def __init__(self, num_features):
        G = 1
        if num_features >= 512:
            G = 32
        elif num_features >= 256:
            G = 16
        elif num_features >= 128:
            G = 8
        elif num_features >= 64:
            G = 4
        self.norm = nn.GroupNorm(num_groups=G, num_channels=num_features)
    def execute(self, x):
        return self.norm(x)

# resnet backbone
class Bottleneck(Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = NormLayer(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = NormLayer(planes * 4)
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
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            NormLayer(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            NormLayer(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = NormLayer(self.inplanes)
        self.relu = nn.ReLU()

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
                NormLayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=1, downsample=downsample))
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
                NormLayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=1,
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

def resnet50(output_stride):
    model = ResNet(Bottleneck, [3,4,6,3], output_stride)
    return model

def resnet101(output_stride):
    model = ResNet(Bottleneck, [3,4,23,3], output_stride)
    return model

# EANet

class ConvBNReLU(Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = NormLayer(c_out)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeconvBNReLU(Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, output_padding, dilation):
        super().__init__()
        self.deconv = nn.ConvTranspose(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, output_padding=output_padding, dilation=dilation, bias=False)
        self.bn = NormLayer(c_out)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class External_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super(External_attention, self).__init__()
        
        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)       
        
        self.conv2 = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), NormLayer(c))        

        self.relu = nn.ReLU()

    def execute(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, n)   # b * c * n 

        attn = self.linear_0(x) # b, k, n

        attn = nn.softmax(attn, dim=-1) # b, k, n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) # b, k, n

        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x

class Cluster_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super().__init__()
        

        self.k = 64
        self.srcconv = nn.Conv(c, self.k, 1, bias=False)
        self.destconv = nn.Conv(c, self.k, 1, bias=False)    

        self.relu = nn.ReLU()

    def execute(self, x):
        b, c, h, w = x.size()

        src = self.srcconv(x) # b, k, h, w
        src = nn.softmax(src, dim=[2, 3]) # b, k, h, w

        read = src.view(b, self.k, 1, h, w) * x.view(b, 1, c, h, w) # b, k, c, h, w
        read = read.sum([3, 4]) # b, k, c

        dest = self.destconv(x)
        dest = nn.softmax(dest, dim=[2, 3]) # b, k, h, w

        write = dest.view(b, self.k, 1, h, w) * read.view(b, self.k, c, 1, 1) # b, k, c, h, w
        write = write.sum(1) # b, c, h, w
        
        x += write
        x = self.relu(x)
        return x



class EANet(Module):
    def __init__(self, num_classes=21, output_stride=16):
        super(EANet, self).__init__()
        self.backbone = resnet101(output_stride)
        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1)
        self.head = External_attention(512)
        self.fc1 = nn.Sequential(
            DeconvBNReLU(512, 256, 3, 2, 1, 1, 1),
            nn.Dropout(p=0.1))
        self.fc2 = nn.ConvTranspose(256, num_classes, kernel_size=2, stride=2, bias=True)

    def execute(self, x):
        imsize = x.shape 
        x = self.backbone(x)
        x = self.fc0(x)
        x = self.head(x) 
        x = self.fc1(x)
        x = self.fc2(x)


        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        return x 

    def get_head(self):
        return [self.fc0, self.head, self.fc1, self.fc2]


def main():
    model = EANet(num_classes=2)
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
    resnet101 backbone
    53,242,818 total parameters.
    53,242,818 training parameters.
    '''

if __name__ == '__main__':
    main()