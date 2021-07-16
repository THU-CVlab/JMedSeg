import jittor as jt
from jittor import init
from jittor import nn

class NormLayer(nn.Module):
    def __init__(self, num_features, num_groups):
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
    
    def execute(self, x):
        return self.norm(x)

class ShiftGroupConv(nn.Module):
    def __init__(self, in_group, out_group, groups, shift, dilation=1, kernel_size=3):
        if kernel_size % 2 != 1:
            raise NotImplementedError
        self.shift = shift
        self.conv = nn.Conv(in_group * groups, out_group * groups,
            kernel_size=kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation, groups=groups, bias=False)
        self.norm = NormLayer(out_group * groups, groups)
        self.act = nn.ReLU()
    def execute(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x_shift = x.roll(shifts=self.shift, dims=1)
        return x_shift

class Input(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size=4, stride=4)

    def execute(self, x):
        return self.conv(x)

class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose(in_channels, out_channels, kernel_size=4, stride=4)

    def execute(self, x):
        return self.conv(x)

class LightNet(nn.Module):
    def __init__(self, n_classes = 2):
        super().__init__()
        self.embed = nn.Sequential(
            Input(1, 64),
            ShiftGroupConv(16, 32, 4, shift=8, dilation=1),
            ShiftGroupConv(16, 32, 8, shift=8, dilation=1),
            ShiftGroupConv(16, 32, 16, shift=8, dilation=1),
            ShiftGroupConv(16, 32, 32, shift=8, dilation=1)
        )
        dilations = [1] * 3 + [2] * 4 + [4] * 6 + [8, 16, 32]
        self.layers = nn.Sequential(
            *[ShiftGroupConv(16, 16, 64, shift=8, dilation=dilations[i]) for i in range(16)]
        )
        self.fuse = nn.Sequential(
            *[ShiftGroupConv(16, 16, 64, shift=8, dilation=1, kernel_size=1) for i in range(7)]
        )
        self.decode = nn.Sequential(
            ShiftGroupConv(32, 16, 32, shift=8, dilation=1, kernel_size=1),
            ShiftGroupConv(32, 16, 16, shift=8, dilation=1, kernel_size=1),
            Output(256, n_classes)
        )

    def execute(self, x):
        x = x.sum(dim=1, keepdims=True) # gray
        x = self.embed(x)
        x_list = [self.layers[0](x)]
        for i in range(1, 16):
            x_list[i:i] = [self.layers[i](x_list[i - 1])]
        fuse_list = x_list[2:3] + x_list[6:7] + x_list[11:13] + x_list[-4:]
        fused = fuse_list[7]
        for i in range(6, -1, -1):
            fused = self.fuse[i](fused * fuse_list[i])
        return self.decode(fused)
        




def main():
    model = LightNet()
    x = jt.ones([2, 3, 512, 512])
    y = model(x)
    print (y.shape)
    _ = y.data

    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    '''
    LightNet
    2,836,802 total parameters.
    2,836,802 training parameters.
    '''

if __name__ == '__main__':
    main()