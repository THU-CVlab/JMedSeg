# Created by Yuanbiao Wang
# The spatial transformer network wrapper for CNNs
# Use forward() to attain forwarding result
# Use stn() to attain warped image


import jittor.nn as nn
import jittor as jt
from jittor.init import constant_
import numpy as np


class STNWrapper(nn.Module):
    def __init__(self, convnet):
        self.convnet = convnet
        self.loc = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        constant_(self.fc_loc[2].weight, 0.)
        constant_(self.fc_loc[2].bias, np.array([1., 0., 0., 0., 1., 0.]))
        
    def stn(self, x):
        xs = self.loc(x)
        xs = xs.view(-1, 256)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = nn.affine_grid(theta, x.size())
        z = nn.grid_sample(x, grid)
        return z
    
    def gen_grid(self, x):
        xs = self.loc(x)
        xs = xs.view(-1, 256)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = nn.affine_grid(theta, x.size())
        return grid

    def execute(self, x):
        z = self.stn(x)
        y = self.convnet(z)
        return y