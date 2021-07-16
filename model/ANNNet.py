import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class PyramidPool(Module):

    def __init__(self,  pool_size):
        super(PyramidPool,self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
    
    def execute(self, x):
        output = self.pool(x)
        return output


class ANNNet(nn.Module):

    def __init__(self, num_classes=2, output_stride=16, backbone = 'resnet101'):
        super(ANNNet, self).__init__()
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
        self.layer5a = PyramidPool(1)
        self.layer5b = PyramidPool(3)
        self.layer5c = PyramidPool(6)
        self.layer5d = PyramidPool(8)

        self.conv_0 = nn.Conv(2048, 2048, kernel_size = 1)
        self.conv_1 = nn.Conv(2048, 2048, kernel_size = 1)
        self.conv_2 = nn.Conv(2048, 2048, kernel_size = 1)
        self.final_conv = nn.Sequential(
        	nn.Conv(2048, 512, 3, padding=1, bias=False),
        	nn.BatchNorm(512),
        	nn.ReLU(),
        	nn.Dropout(.3),
        	nn.Conv(512, num_classes, 1),
        )

    def execute(self, x):
        imsize = x.shape
        _, _, _, x = self.backbone(x)
        b, c, h, w = x.shape
        x_k = self.conv_0 (x)
        x1 = self.layer5a(x_k).reshape(b, c, -1)
        x2 = self.layer5b(x_k).reshape(b, c, -1)
        x3 = self.layer5c(x_k).reshape(b, c, -1)
        x4 = self.layer5d(x_k).reshape(b, c, -1)
        x_k = concat ([x1, x2, x3, x4], 2).transpose(0, 2, 1) # b  110 c
        x_q = self.conv_1(x)
        x_q = x_q.reshape(b, c, -1)
        x_attention = nn.bmm(x_k, x_q) # b 110 N 
        x_v = self.conv_2 (x)
        x1 = self.layer5a(x_v).reshape(b, c, -1)
        x2 = self.layer5b(x_v).reshape(b, c, -1)
        x3 = self.layer5c(x_v).reshape(b, c, -1)
        x4 = self.layer5d(x_v).reshape(b, c, -1)
        x_v = concat ([x1, x2, x3, x4], 2) # b c 110
        x = nn.bmm(x_v, x_attention).reshape(b, c, h, w)
        x = self.final_conv(x)
        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear', align_corners=True) 
        return x


    def get_backbone(self):
        return self.backbone
    def get_head(self):
        return [self.conv_0, self.conv_1, self.conv_2, self.final_conv]
    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index)
        return loss_pred
    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    model = ANNNet(backbone = 'resnet101')
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()