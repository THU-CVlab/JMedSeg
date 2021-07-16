import jittor as jt
from jittor import nn
from jittor import Module
from jittor import init
from jittor.contrib import concat


# https://github.com/Jittor/segmentation-jittor/blob/master/backbone/mobilenet.py

def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv(inp, oup, 3, stride, 1, bias=False),
        BatchNorm(oup),
        nn.ReLU6()
    )


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padd_func = nn.ZeroPad2d((pad_beg, pad_end, pad_beg, pad_end))
    padded_inputs = padd_func(inputs)
    return padded_inputs


class InvertedResidual(Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv(hidden_dim, oup, 1, 1, 0, 1, 1, bias=False),
                BatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv(inp, hidden_dim, 1, 1, 0, 1, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(),
                # dw
                nn.Conv(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv(hidden_dim, oup, 1, 1, 0, 1, bias=False),
                BatchNorm(oup),
            )

    def execute(self, x):
        # changed
        x_pad = fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            x = x + self.conv(x_pad)
        else:
            x = self.conv(x_pad)

        return x


class MobileNetV2(Module):
    def __init__(self, output_stride=8, BatchNorm=None, width_mult=1., pretrained=True):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        current_stride = 1
        rate = 1
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.features = [conv_bn(3, input_channel, 2, BatchNorm)]
        current_stride *= 2
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride, dilation, t, BatchNorm))
                else:
                    self.features.append(block(input_channel, output_channel, 1, dilation, t, BatchNorm))
                input_channel = output_channel
        # self.features = nn.Sequential(*self.features)
        # self._initialize_weights()

        # if pretrained:
        #     self._load_pretrained_model()

        self.low_level_features = self.features[0:4]
        self.low_level_features = nn.Sequential(*self.low_level_features)
        self.high_level_features = self.features[4:]
        self.high_level_features = nn.Sequential(*self.high_level_features)

    def execute(self, x):
        #print (type(x))
        #print (x.shape)
        low_level_feat = self.low_level_features(x)
        x = self.high_level_features(low_level_feat)
        return x, low_level_feat

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

def mobilenet(output_stride):
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm)
    params_path = 'model/pretrained_models/mobilenet_v2-6a65762b.pth'
    load_pretrained_model (model, params_path)   
    return model 
    # def _load_pretrained_model(self):
    #     pretrain_dict = model_zoo.load_url('http://jeff95.me/models/mobilenet_v2-6a65762b.pth')
    #     model_dict = {}
    #     state_dict = self.state_dict()
    #     for k, v in pretrain_dict.items():
    #         if k in state_dict:
    #             model_dict[k] = v
    #     state_dict.update(model_dict)
    #     self.load_state_dict(state_dict)

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             # m.weight.data.normal_(0, math.sqrt(2. / n))
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, SynchronizedBatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

if __name__ == "__main__":
    input = jt.random((2, 3, 512, 512)) # must add ()
    print (type(input))
    model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm)
    output, low_level_feat = model(input)
    print(output.data.shape)
    print(low_level_feat.data.shape)