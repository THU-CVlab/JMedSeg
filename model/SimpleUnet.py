import jittor as jt
from jittor import init
from jittor import nn

class NormLayer(nn.Module):
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

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            NormLayer(out_channels),
            nn.ReLU()
        )

    def execute(self, x):
        x = self.block(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Pool(2, op='maximum'), 
            SimpleBlock(in_channels, out_channels)
        )

    def execute(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, shortcut_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose(in_channels, shortcut_channels, kernel_size=2, stride=2, bias=False)
        self.conv = SimpleBlock(shortcut_channels, out_channels)

    def execute(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(x1 * x2)

class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size=2, stride=2)

    def execute(self, x):
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose(in_channels, out_channels, kernel_size=2, stride=2)

    def execute(self, x):
        return self.conv(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = InConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.outc = OutConv(64, n_classes)

    def execute(self, x):
        x2 = self.inc(x)
        x4 = self.down1(x2)
        x8 = self.down2(x4)
        x16 = self.down3(x8)
        y8 = self.up1(x16, x8)
        y4 = self.up2(y8, x4)
        y2 = self.up3(y4, x2)
        logits = self.outc(y2)
        return logits

def main():
    model = SimpleUNet()
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
    SimpleUNet
    7,082,178 total parameters.
    7,082,178 training parameters.
    '''

if __name__ == '__main__':
    main()