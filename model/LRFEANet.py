import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat

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

# Large Receptive Field resnet backbone
class PatchMergeBlock(Module):
    def __init__(self, in_dim, out_dim, patch_size):
        super().__init__()
        expansion = patch_size * patch_size
        self.block=nn.Sequential(
            nn.Conv(in_dim, in_dim * expansion, kernel_size=patch_size, stride=patch_size, bias=False),
            NormLayer(in_dim * expansion),
            nn.ReLU(),
            nn.Conv(in_dim * expansion, out_dim, 1, bias=False),
            NormLayer(out_dim),
            nn.ReLU())
    
    def execute(self, x):
        return self.block(x)

class PatchSplitBlock(Module):
    def __init__(self, in_dim, out_dim, patch_size, expansion=None):
        super().__init__()
        if expansion == None:
            expansion = patch_size * patch_size
        self.block=nn.Sequential(
            nn.Conv(in_dim, in_dim * expansion, 1, bias=False),
            NormLayer(in_dim * expansion),
            nn.ReLU(),
            nn.ConvTranspose(in_dim * expansion, out_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            NormLayer(out_dim),
            nn.ReLU())    
    
    def execute(self, x):
        return self.block(x)

class AggregationBlock(Module):
    def __init__(self, dim, dilation=1, expansion=2):
        super().__init__()
        self.nonlinearity = nn.ReLU()
        self.block = self._make_block(dim, dilation, expansion)
    def _make_block(self, c, d, e):
        block = []        
        block.append(nn.Conv(c, c * e, 3, padding=d, dilation=d, bias=False))
        block.append(NormLayer(c * e))
        block.append(self.nonlinearity) # nonlinear neighbor aggregate
        block.append(nn.Conv(c * e, c, 1, bias=False))
        block.append(NormLayer(c))
        return nn.Sequential(*block)

    def execute(self, x):
        identity = x

        out = self.block(x)

        out += identity
        out = self.nonlinearity(out)

        return out

class LRFResNet(Module):
    def __init__(self, layers, output_stride):
        super().__init__()

        if output_stride != 16:
            raise NotImplementedError

        modules = []

        # patch merge
        modules.append(PatchMergeBlock(3, 64, patch_size=2))
        modules.append(AggregationBlock(64))
        modules.append(PatchMergeBlock(64, 64, patch_size=2))

        # layer1
        modules.append(self._make_layer(64, dilations=layers[0]))

        # patch merge
        modules.append(PatchMergeBlock(64, 128, patch_size=2))

        # layer2
        modules.append(self._make_layer(128, dilations=layers[1]))

        # patch merge
        modules.append(PatchMergeBlock(128, 256, patch_size=2))
        
        # layer3
        modules.append(self._make_layer(256, dilations=layers[2]))

        modules.append(nn.Conv(256, 512, 1, bias=False))
        modules.append(NormLayer(512))
        modules.append(nn.ReLU())


        # layer4
        modules.append(self._make_layer(512, dilations=layers[3]))

        self.net = nn.Sequential(*modules)


    def _make_layer(self, feature_dim, dilations):
        layers = []
        for d in dilations:
            layers.append(AggregationBlock(feature_dim, dilation=d))
        return nn.Sequential(*layers)

    def execute(self, x):
        return self.net(x)

def LRFresnetS(output_stride):
    model = LRFResNet([[1] * 5, [1] * 5, [1, 1, 2] * 6, [1, 2, 4, 8] * 1], output_stride)
    return model

# EANet
class External_attention(Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''
    def __init__(self, c):
        super().__init__()
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
        x = x.view(b, c, h*w)   # b * c * n 

        attn = self.linear_0(x) # b, k, n

        attn = nn.softmax(attn, dim=-1) # b, k, n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) # b, k, n

        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x



class LRFEANet(Module):
    def __init__(self, num_classes=2, output_stride=16):
        super().__init__()
        self.backbone = LRFresnetS(output_stride)
        
        self.head = External_attention(512)

        self.split = nn.Sequential(
            PatchSplitBlock(512, 256, patch_size=2),
            nn.Dropout(p=0.1))
            
        self.fc = nn.ConvTranspose(256, num_classes, kernel_size=2, stride=2, bias=True)
        

    def execute(self, x):
        imsize = x.shape
        x = self.backbone(x)

        x = self.head(x)

        x = self.split(x)

        x = self.fc(x)

        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        return x 

    def get_head(self):
        return [self.fc0, self.head, self.fc1, self.fc2]


def main():
    model = LRFEANet(num_classes=2)
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
    resnet50 backbone
    34,773,954 total parameters.
    34,718,274 training parameters.
    '''

if __name__ == '__main__':
    main()