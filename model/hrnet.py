import jittor as jt
from jittor import init
from jittor import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv(planes, (planes * self.expansion), 1, bias=False)
        self.bn3 = nn.BatchNorm((planes * self.expansion), momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(planes, momentum=bn_momentum)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv(inplanes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if (self.downsample is not None):
            residual = self.downsample(x)
        out += residual
        out = nn.relu(out)
        return out

class StageModule(nn.Module):

    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = (c * (2 ** i))
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum), 
                BasicBlock(w, w, bn_momentum=bn_momentum), 
                BasicBlock(w, w, bn_momentum=bn_momentum), 
                BasicBlock(w, w, bn_momentum=bn_momentum)
            )
            self.branches.append(branch)
        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):
                if (i == j):
                    self.fuse_layers[(- 1)].append(nn.Sequential())
                elif (i < j):
                    self.fuse_layers[(- 1)].append(
                        nn.Sequential(
                            nn.Conv((c * (2 ** j)), (c * (2 ** i)), (1, 1), stride=(1, 1), bias=False), 
                            nn.BatchNorm((c * (2 ** i)), eps=1e-05, momentum=0.1, affine=True), 
                            nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest')
                        )
                    )
                elif (i > j):
                    ops = []
                    for k in range(((i - j) - 1)):
                        ops.append(
                            nn.Sequential(
                                nn.Conv((c * (2 ** j)), (c * (2 ** j)), (3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                                nn.BatchNorm((c * (2 ** j)), eps=1e-05, momentum=0.1, affine=True), 
                                nn.ReLU()
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv((c * (2 ** j)), (c * (2 ** i)), (3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                            nn.BatchNorm((c * (2 ** i)), eps=1e-05, momentum=0.1, affine=True)
                        )
                    )
                    self.fuse_layers[(- 1)].append(nn.Sequential(*ops))
        self.relu = nn.ReLU()

    def execute(self, x):
        assert (len(self.branches) == len(x))
        x = [branch(b) for (branch, b) in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if (j == 0):
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = (x_fused[i] + self.fuse_layers[i][j](x[j]))
        for i in range(len(x_fused)):
            x_fused[i] = nn.relu(x_fused[i])
        return x_fused

class HRNet(nn.Module):

    def __init__(self, c=16, in_ch=3, out_ch=2, bn_momentum=0.1):
        super(HRNet, self).__init__()
        self.conv1 = nn.Conv(in_ch, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm(64, eps=1e-05, momentum=bn_momentum, affine=True)
        self.conv2 = nn.Conv(64, 64, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm(64, eps=1e-05, momentum=bn_momentum, affine=True)
        self.relu = nn.ReLU()
        downsample = nn.Sequential(
            nn.Conv(64, 256, (1, 1), stride=(1, 1), bias=False), 
            nn.BatchNorm(256, eps=1e-05, momentum=bn_momentum, affine=True)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample), 
            Bottleneck(256, 64), 
            Bottleneck(256, 64), 
            Bottleneck(256, 64)
        )
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv(256, c, (3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
                nn.BatchNorm(c, eps=1e-05, momentum=bn_momentum, affine=True), 
                nn.ReLU()
            ), 
            nn.Sequential(nn.Sequential(
                nn.Conv(256, (c * (2 ** 1)), (3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                nn.BatchNorm((c * (2 ** 1)), eps=1e-05, momentum=bn_momentum, affine=True), 
                nn.ReLU())
            )
        ])
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum)
        )
        self.transition2 = nn.ModuleList([
            nn.Sequential(), 
            nn.Sequential(), 
            nn.Sequential(nn.Sequential(
                nn.Conv((c * (2 ** 1)), (c * (2 ** 2)), (3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                nn.BatchNorm((c * (2 ** 2)), eps=1e-05, momentum=bn_momentum, affine=True), 
                nn.ReLU())
            )
        ])
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), 
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), 
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum), 
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum)
        )
        self.transition3 = nn.ModuleList([
            nn.Sequential(), 
            nn.Sequential(), 
            nn.Sequential(), 
            nn.Sequential(nn.Sequential(
                nn.Conv((c * (2 ** 2)), (c * (2 ** 3)), (3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                nn.BatchNorm((c * (2 ** 3)), eps=1e-05, momentum=bn_momentum, affine=True), 
                nn.ReLU())
            )
        ])
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), 
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum), 
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum)
        )
        self.final_layer = nn.Conv(c, out_ch, (1, 1), stride=(1, 1))

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.relu(x)
        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]
        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]), 
            self.transition2[1](x[1]), 
            self.transition2[2](x[(- 1)])
        ]
        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]), 
            self.transition3[1](x[1]), 
            self.transition3[2](x[2]), 
            self.transition3[3](x[(- 1)])
        ]
        x = self.stage4(x)
        x = self.final_layer(x[0])
        return x

