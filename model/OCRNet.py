import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class OCRHead(Module):
    def __init__(self, in_channels, n_cls):
        super(OCRHead, self).__init__() 
        self.relu = nn.ReLU() 
        self.in_channels = in_channels
        self.softmax = nn.Softmax(dim = 2)
        self.conv_1x1 = nn.Conv(in_channels, in_channels, kernel_size=1)
        self.last_conv = nn.Conv(in_channels * 2, n_cls, kernel_size=3, stride=1, padding=1)
        self._zero_init_conv() 
    def _zero_init_conv(self):
        self.conv_1x1.weight = init.constant([self.in_channels, self.in_channels, 1, 1], 'float', value=0.0)

    def execute(self, context, feature):
        batch_size, c, h, w = feature.shape 
        origin_feature = feature 
        feature = feature.reshape(batch_size, c, -1).transpose(0, 2, 1) # b, h*w, c
        context = context.reshape(batch_size, context.shape[1], -1) # b, n_cls, h*w
        attention = self.softmax(context)
        ocr_context = nn.bmm(attention, feature).transpose(0, 2, 1) # b, c, n_cls
        relation = nn.bmm(feature, ocr_context).transpose(0, 2, 1) # b, n_cls, h*w
        attention = self.softmax(relation) #b , n_cls, h*w 
        result = nn.bmm(ocr_context, attention).reshape(batch_size, c, h, w) 
        result = self.conv_1x1(result)
        result = concat ([result, origin_feature], dim=1)
        result = self.last_conv (result)
        return result
        

class OCRNet(Module):
    def __init__(self, num_classes=2, output_stride=16, backbone = 'resnet101'):
        super(OCRNet, self).__init__()
        self.num_classes = num_classes 
        in_channels = [1024, 2048]

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

        # self.backbone = resnet101(output_stride)
        self.head = OCRHead(512, num_classes)
        self.get_context = nn.Sequential(
            nn.Conv(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(512),
            nn.ReLU(),
         )


    def execute(self, x):
        imsize = x.shape
        x_, x__, x_0, x = self.backbone(x)
        ## begin ocrhead
        context = self.get_context(x_0)
        x_feature = self.conv_3x3 (x)
        

        x = self.head(context, x_feature)
        x = nn.resize(x, size=(imsize[2], imsize[3]), mode='bilinear')
        context = nn.resize(context, size=(imsize[2], imsize[3]), mode='bilinear')
    
        return x
    
    def get_backbone(self):
        return self.backbone

    def get_head(self):
        return [self.get_context, self.conv_3x3, self.head]

    def get_loss(self, target, pred, context=None, ignore_index=None):
        
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index)
        if context is None:
            return loss_pred
        loss_context = nn.cross_entropy_loss(context, target, ignore_index=ignore_index)
        loss = loss_pred + 0.1 * loss_context
        return loss

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def main():
    model = OCRNet(backbone = 'resnet50')
    x = jt.ones([2, 3, 512, 512])
    context, y = model(x)
    print (context.shape, y.shape)
    _ = y.data

if __name__ == '__main__':
    main()