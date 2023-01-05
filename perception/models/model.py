import torch
import torch.nn as nn
from .layers import BaseConv, Residual, SPPBlock

class Darknet53(nn.Module):
    '''
    Base Darknet53 based on the YOLOv3 paper
    Args:
        in_channels (int): number of input channels
        stem_out_channels (int): number of output channels for first convolutional layer
        output (tuple): output layers to return
    '''
    def __init__(self, in_channels: int, stem_out_channels: int = 32, output: tuple = ('c3', 'c4', 'c5')):
        super().__init__()
        self.output = output
        num_blocks = [2, 8, 8, 4]
        self.c1 = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, kernel_size=3, stride=1, activation='lrelu'),
            *self._build_group_block(in_channels=stem_out_channels, num_blocks=1, stride=2))
        in_channels = stem_out_channels * 2
        self.c2 = nn.Sequential(
            *self._build_group_block(in_channels=in_channels, num_blocks=num_blocks[0], stride=2))
        in_channels = in_channels * 2
        self.c3 = nn.Sequential(
            *self._build_group_block(in_channels=in_channels, num_blocks=num_blocks[1], stride=2))
        in_channels = in_channels * 2
        self.c4 = nn.Sequential(
            *self._build_group_block(in_channels=in_channels, num_blocks=num_blocks[2], stride=2))
        in_channels = in_channels * 2
        self.c5 = nn.Sequential(
            *self._build_group_block(in_channels=in_channels, num_blocks=num_blocks[3], stride=2))

    def _build_group_block(self, in_channels: int, num_blocks: int, stride: int):
        '''
        Build convolutional layer -> Residual Block (repeated num_blocks times)
        '''
        return [
            BaseConv(in_channels, in_channels*2, kernel_size=3, stride=stride),
            *[(Residual(in_channels*2)) for _ in range(num_blocks)]
        ]

    def forward(self, x):
        outputs = {}
        x = self.c1(x)
        outputs['c1'] = x
        x = self.c2(x)
        outputs['c2'] = x
        x = self.c3(x)
        outputs['c3'] = x
        x = self.c4(x)
        outputs['c4'] = x
        x = self.c5(x)
        outputs['c5'] = x
        return {k:v for k, v in outputs.items() if k in self.output}

class SPP(nn.Module):
    '''
    Spatial Pyramid Pooling block

    '''
    def __init__(self, in_channels, ):
        super().__init__()
        self.spp = nn.Sequential()

    def forward(self, x):
        pass


class FPN(nn.Module):
    '''
    Feature Pyramid Network
        - merge shallow layers with deeper layers
    Args:

    '''
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, ):
        pass

class YOLOHead(nn.Module):
    def __init__(self):
        pass

    def forward(self, c3, c4, c5):
        pass
