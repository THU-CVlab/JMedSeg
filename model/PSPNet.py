import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat
from model.backbone import resnet50, resnet101
from model.backbone import res2net101

Backbone_List = ['resnet50', 'resnet101', 'res2net101']

class PyramidPool(Module):

	def __init__(self, in_channels, out_channels, pool_size):
		super(PyramidPool,self).__init__()

		self.conv = nn.Sequential(
			nn.AdaptiveAvgPool2d(pool_size),
			nn.Conv(in_channels, out_channels, 1, bias=False),
			nn.BatchNorm(out_channels),
			nn.ReLU()
		)


	def execute(self, x):
		size = x.shape
		output = nn.resize(self.conv(x), size=(size[2], size[3]), mode='bilinear')
		return output


class PSPNet(nn.Module):

    def __init__(self, num_classes=2, output_stride=16, backbone = 'resnet101'):
        super(PSPNet,self).__init__()
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

        self.layer5a = PyramidPool(2048, 512, 1)
        self.layer5b = PyramidPool(2048, 512, 2)
        self.layer5c = PyramidPool(2048, 512, 3)
        self.layer5d = PyramidPool(2048, 512, 6)

        self.final_conv = nn.Sequential(
        	nn.Conv(4096, 512, 3, padding=1, bias=False),
        	nn.BatchNorm(512),
        	nn.ReLU(),
        	nn.Dropout(.3),
        	nn.Conv(512, num_classes, 1),
        )

    def execute(self, x):
        size = x.shape
        _, _, _, x = self.backbone(x)
        x = self.final_conv(concat([
        	x,
        	self.layer5a(x),
        	self.layer5b(x),
        	self.layer5c(x),
        	self.layer5d(x),
        ], 1))

        return nn.resize(x, size=(size[2], size[3]), mode='bilinear')

    def get_backbone(self):
        return self.backbone
    def get_head(self):
        return [self.layer5a, self.layer5b, self.layer5c, self.layer5d, self.final_conv]

    def get_loss(self, target, pred, ignore_index=None):
        loss_pred = nn.cross_entropy_loss(pred, target, ignore_index=ignore_index)
        return loss_pred

    def update_params(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


def main():
    model = PSPNet(backbone = 'resnet101')
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

if __name__ == '__main__':
    main()