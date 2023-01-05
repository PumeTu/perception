import torch
import torch.nn as nn
from layers import BaseConv, Residual, SPPBlock
from backbone.darknet53 import Darknet53
from neck.fpn import FPN
from head.heads import YOLOHead, SegmentationHead

class Perception(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, x):
        pass