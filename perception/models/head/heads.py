import torch
import torch.nn as nn
from typing import Dict, List
from ..layers import BaseConv
from utils.utils import iou

class YOLOHead(nn.Module):
    def __init__(self, in_channels: tuple = (512, 256, 128), num_classes: int = 80, num_anchors: int = 3)       
        super().__init__()
        self.conv1 = nn.Sequential(
            BaseConv(in_channels[0], 2 * in_channels[0], kernel_size=3, stride=1, activation='lrelu'),
            nn.Conv2d(in_channels[0] * 2, num_anchors * (num_classes + 5), kernel_size=1, stride=1)
        )
        self.conv2 = nn.Sequential(
            BaseConv(in_channels[1], 2 * in_channels[1], kernel_size=3, stride=1, activation='lrelu'),
            nn.Conv2d(in_channels[1] * 2, num_anchors * (num_classes + 5), kernel_size=1, stride=1)
        )
        self.conv3 = nn.Sequential(
            BaseConv(in_channels[2], 2 * in_channels[2], kernel_size=3, stride=1, activation='lrelu'),
            nn.Conv2d(in_channels[2] * 2, num_anchors * (num_classes + 5), kernel_size=1, stride=1)
        )
    def forward(self, p5, p4, p3):
        det1 = self.conv1(p5)
        det2 = self.conv2(p4)
        det3 = self.conv3(p3)
        return det1, det2, det3

class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.m = nn.Sequential(
            BaseConv(in_channels, in_channels / 2, kernel_size=3, stride=1, activation='lrelu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BaseConv(in_channels / 4, in_channels, in_channels / 8, kernel_size=3, stride=1, activation='lrelu'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BaseConv(in_channels / 8, in_channels, in_channels / 16, kernel_size=3, stride=1, activation='lrelu'),
            nn.Upsample(scale_factor=2, mode='nearest'), 
            BaseConv(in_channels / 16, in_channels, num_classes, kernel_size=3, stride=1, activation='lrelu'),
        )

    def forward(self, x):
        return self.m(x)

