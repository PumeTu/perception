import torch
import torch.nn as nn

def iou(pred:torch.Tensor, target: torch.Tensor, format:str = 'xyxy', type: str = None):
    """
    Caculate IoU between 2 boxes
    Args:
        pred (Tensor)
        target (Tensor)
        format (str)
        type (str)
    """
    assert pred.shape[0] == target.shape[0]
    pred = pred.view(-1, 4)
    target = target.view(-1, 4)
