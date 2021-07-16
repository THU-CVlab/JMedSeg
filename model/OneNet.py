import jittor as jt
from jittor import init
from jittor import nn

class NormLayer(nn.Module):
    def __init__(self, num_features, num_groups=None):
        if num_groups is None:
            num_groups = 1
            if num_features >= 512:
                num_groups = 32
            elif num_features >= 256:
                num_groups = 16
            elif num_features >= 128:
                num_groups = 8
            elif num_features >= 64:
                num_groups = 4
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    
    def execute(self, x):
        return self.norm(x)

class ShiftGroupConv(nn.Module):
    def __init__(self, in_group, out_group, groups, dilation=1):
        self.shift = out_group // 2
        self.conv = nn.Conv(in_group * groups, out_group * groups,
            kernel_size=3, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.norm = NormLayer(out_group * groups, groups)
        self.act = nn.ReLU()
    def execute(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x_shift = x.roll(shifts=self.shift, dims=1)
        return x_shift

class In(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size=4, stride=4)

    def execute(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose(in_channels, out_channels, kernel_size=4, stride=4)

    def execute(self, x):
        return self.conv(x)

class OneNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 2):
        super().__init__()
        self.embed = In(n_channels, 128)
        self.map = nn.Sequential(
            ShiftGroupConv(16, 32, 8, dilation=1),
            ShiftGroupConv(16, 32, 16, dilation=1),
            ShiftGroupConv(16, 32, 32, dilation=1)
        )
        dilations = [1] * 3 + [2] * 4 + [4] * 6 + [8, 16, 32]
        self.layers = nn.Sequential(
            *[ShiftGroupConv(16, 16, 64, dilation=dilations[i]) for i in range(16)]
        )
        self.fuse = nn.Sequential(
            nn.Conv(8192, 256, kernel_size=1, bias=False),
            NormLayer(256, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            OutConv(256, n_classes)
        )

    def execute(self, x):
        x = self.embed(x)
        x = self.map(x)
        x_list = [self.layers[0](x)]
        for i in range(1, 16):
            x_list[i:i] = [self.layers[i](x_list[i - 1])]
        return self.fuse(jt.contrib.concat(x_list[2:3] + x_list[6:7] + x_list[11:13] + x_list[-4:], dim=1))



def main():
    model = OneNet()
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
    OneNet
    4,765,826 total parameters.
    4,765,826 training parameters.
    '''

if __name__ == '__main__':
    main()