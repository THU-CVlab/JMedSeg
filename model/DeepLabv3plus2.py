import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class DeepLab(Module):
    def __init__(self, output_stride=16, num_classes=2, backbone = 'resnet101'):
        super(DeepLab, self).__init__()
        if not backbone in Backbone_List:
            print('Invalid Backbone! Initialized to resnet101')
            backbone = 'resnet101'
        if backbone == 'resnet50':
            self.backbone = resnet50(output_stride=output_stride)
        elif backbone == 'res2net101':
            self.backbone = res2net101(output_stride=output_stride)
        else:
            self.backbone = resnet101(output_stride=output_stride)

        self.backbone_name = backbone
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(num_classes)

    def execute(self, input):

        low_level_feat, _, _, x = self.backbone(input)

        x = self.aspp(x)

        x = self.decoder(x, low_level_feat)

        x = nn.resize(x, size=(input.shape[2], input.shape[3]), mode='bilinear')

        return x

    def get_backbone(self):
        return self.backbone
    def get_head(self):
        return [self.aspp, self.decoder]
    
    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        low_level_inplanes = 256 # mobilenet = 24 resnet / res2net = 256 xception = 128

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
                                       nn.Conv(256, num_classes, kernel_size=1, stride=1))


    def execute(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        #print (low_level_feat.shape)
        x = nn.resize(x, size=(low_level_feat.shape[2], low_level_feat.shape[3]) , mode='bilinear')
        x = concat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


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
        inplanes = 2048 # mobilnet = 320 resnet = 2048 
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

def main():
    model = DeepLab(backbone = 'resnet101')
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
    DeepLab
    59,572,610 total parameters.
    59,462,946 training parameters.
    '''

if __name__ == '__main__':
    main()