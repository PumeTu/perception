import torch
import torch.nn as nn
from ..layers import BaseConv

class FPN(nn.Module):
    '''
    Feature Pyramid Network
        - merge shallow layers with deeper layers
    Args:

    '''
    def __init__(self, input: tuple = ('c3', 'c4', 'c5'), output: tuple = ('p3', 'p4', 'p5')):
        super().__init__()
        self.input = input
        self.output = output
        self.conv1 = BaseConv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, activation="lrelu")
        self.conv2 = BaseConv(in_channels=512, out_channels=256, kernel_size=1, stride=1, activation="lrelu")
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        outputs = {}
        c3, c4, c5 = [x[i] for i in self.input]
        outputs['p5'] = c5
        #P4
        m4 = self.conv1(c5)
        m4 = self.upsample(m4)
        outputs['p4'] = torch.cat([m4, c4], dim=1)
        #P3
        m3 = self.conv2(c4)
        m3 = self.upsample(m3)
        outputs['p3'] = torch.cat([m3, c3], dim=1)

        return {k:v for k, v in outputs.items() if k in self.output}