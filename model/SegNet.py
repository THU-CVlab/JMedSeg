from jittor import nn

class SegNet(nn.Module):
    def __init__(self, input_nbr, label_nbr):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)


    def execute(self, x):
        # Stage 1
        x11 = nn.relu(self.bn11(self.conv11(x)))
        x12 = nn.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = self.max_pool2d(x12)

        # Stage 2
        x21 = nn.relu(self.bn21(self.conv21(x1p)))
        x22 = nn.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = self.max_pool2d(x22)

        # Stage 3
        x31 = nn.relu(self.bn31(self.conv31(x2p)))
        x32 = nn.relu(self.bn32(self.conv32(x31)))
        x33 = nn.relu(self.bn33(self.conv33(x32)))
        x3p, id3 = self.max_pool2d(x33)

        # Stage 4
        x41 = nn.relu(self.bn41(self.conv41(x3p)))
        x42 = nn.relu(self.bn42(self.conv42(x41)))
        x43 = nn.relu(self.bn43(self.conv43(x42)))
        x4p, id4 = self.max_pool2d(x43)

        # Stage 5
        x51 = nn.relu(self.bn51(self.conv51(x4p)))
        x52 = nn.relu(self.bn52(self.conv52(x51)))
        x53 = nn.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = self.max_pool2d(x53)


        # Stage 5d
        x5d = self.max_unpool2d(x5p, id5)
        x53d = nn.relu(self.bn53d(self.conv53d(x5d)))
        x52d = nn.relu(self.bn52d(self.conv52d(x53d)))
        x51d = nn.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.max_unpool2d(x51d, id4)
        x43d = nn.relu(self.bn43d(self.conv43d(x4d)))
        x42d = nn.relu(self.bn42d(self.conv42d(x43d)))
        x41d = nn.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.max_unpool2d(x41d, id3)
        x33d = nn.relu(self.bn33d(self.conv33d(x3d)))
        x32d = nn.relu(self.bn32d(self.conv32d(x33d)))
        x31d = nn.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.max_unpool2d(x31d, id2)
        x22d = nn.relu(self.bn22d(self.conv22d(x2d)))
        x21d = nn.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.max_unpool2d(x21d, id1)
        x12d = nn.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
