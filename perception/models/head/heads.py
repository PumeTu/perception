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
    def __init__(self, in_channels: tuple = (128, 256, 512), num_classes: int = 20, anchors: tuple = ()):
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5
        self.anchors = anchors
        self.num_anchors = len(anchors[0]) // 2
        self.num_detections = len(anchors)
        self.register_buffer('anchors', anchors)
        self.register_buffer()
        self.convs = nn.ModuleList(nn.Conv2d(in_channel, self.num_outputs * self.num_anchors, kernel_size=1) for in_channel in in_channels)

    def foward(self, x):
        """
        Forward pass of the network to predict outputs given the three layers from the FPN
        Note:
            There are differnet outputs during training and inference
            - During training we are given the index where the anchors contains object thus in the forward pass
                we only return those with objects
            - During inference we do not have the labels thus we return all predictions

        Return (Training):
            - 

        Return (Inference):
            - 
        
        """
        for i in range(self.num_detections):
            x[i] = self.convs[i](x[i])
            bs, _, h, w = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, h, w).permute(0, 1, 3, 4, 2).contiguous()




class SegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

