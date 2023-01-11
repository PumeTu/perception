import torch
import torch.nn as nn
from layers import BaseConv, Residual, SPPBlock
from backbone.darknet53 import Darknet53
from neck.fpn import FPN
from head.heads import YOLOHead, SegmentationHead

class Perception(nn.Module):
    """
    
    """
    def __init__(self, in_channels: int, num_det: int, num_lane: int, num_drivable: int, num_anchors: int):     
        super().__init__()
        self.backbone = Darknet53(in_channels=in_channels)
        self.fpn = FPN()
        self.detection_head = YOLOHead(num_classes=num_det, num_anchors=num_anchors)
        self.lane_head = SegmentationHead(in_channels=128, num_classes=num_lane)
        self.drivable_head = SegmentationHead(in_channels=128, num_classes=num_drivable)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        detection = self.detection_head(x)
        lane = self.lane_head(x[0])
        drivable = self.drivable_head(x[0])

        return detection, lane, drivable