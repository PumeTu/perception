import torch
import torch.nn as nn
from typing import Dict, List
from ..layers import BaseConv
from utils.utils import iou

def generate_anchors():
    pass

def generate_proposals():
    pass

class YOLOHead(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_anchors, num_classes):
        super().__init__()
        assert(num_classes != 0 and num_anchors != 0)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.head = torch.nn.Sequential(
            BaseConv(in_channels, hidden_dim, kernel_size=1, stride=1, activation='lrelu'),
            BaseConv(hidden_dim, int(5 * num_anchors + num_classes), kernel_size=1, stirde=1, activation='lrelu')
        )



class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

