import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class DANet(Module):
    def __init__(self, num_classes=2, output_stride=16, backbone = 'resnet101'):
        super(DANet, self).__init__()
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
        self.head = DANetHead(2048, num_classes)

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
    def get_loss(self, target, pred, context=None, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index)
        return loss_pred 
    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class DANetHead(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm(inter_channels),
                                   nn.ReLU())

#        self.conv6 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv(inter_channels, out_channels, 1))
#        self.conv7 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv(inter_channels, out_channels, 1))

    def execute(self, x):

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = jt.zeros(1)

        self.softmax = nn.Softmax(dim=-1)
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

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = jt.zeros(1)
        self.softmax  = nn.Softmax(dim=-1)
    def execute(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.reshape(m_batchsize, C, -1)
        proj_key = x.reshape(m_batchsize, C, -1).transpose(0, 2, 1)
        energy = nn.bmm(proj_query, proj_key)
        #energy_new = jt.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy)
        proj_value = x.reshape(m_batchsize, C, -1)

        out = nn.bmm(attention, proj_value)
        out = out.reshape(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




def main():
    model = DANet(backbone = 'resnet50')
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()