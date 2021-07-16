import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class OCNet(Module):
    def __init__(self, num_classes=2, output_stride=16, backbone = 'resnet101'):
        super(OCNet, self).__init__()
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

        self.head = OCHead(2048, num_classes)

    def execute(self, x):
        imsize = x.shape
        _, _, _, x = self.backbone(x)

        x = self.head(x)
        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        return x
    
    def get_backbone(self):
        return self.backbone

    def get_head(self):
        return [self.head]

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index) 
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class OCHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OCHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_0 = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())

        self.oc = OC_Module(inter_channels)
        self.conv_1 = nn.Sequential(nn.Conv(inter_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv(inter_channels, out_channels, 1))

    def execute(self, x):

        feat1 = self.conv_0(x)
        oc_feat = self.oc(feat1)
        oc_conv = self.conv_1(oc_feat)
        oc_output = self.conv_2(oc_conv)

        return oc_output


class OC_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(OC_Module, self).__init__()
        self.in_channels = in_dim

        self.query_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.scale_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self._zero_init_conv() 
    def _zero_init_conv(self):
        self.scale_conv.weight = init.constant([self.in_channels, self.in_channels, 1, 1], 'float', value=0.0)
    def execute(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).reshape(m_batchsize, -1, width*height).transpose(0, 2, 1)
        proj_key = self.key_conv(x).reshape(m_batchsize, -1, width*height)
        energy = nn.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).reshape(m_batchsize, -1, width*height)

        out = nn.bmm(proj_value, attention.transpose(0, 2, 1))
        out = out.reshape(m_batchsize, C, height, width)
        out = self.scale_conv(out)
        out = concat([out, x], 1)
        return out



def main():
    model = OCNet(backbone = 'resnet50')
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()