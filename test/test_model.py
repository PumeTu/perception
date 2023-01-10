import unittest
import torch
from perception.models.backbone.darknet53 import Darknet53
from perception.models.neck.fpn import FPN

class TestModel(unittest.TestCase):
    def test_darknet53(self):
        input_width, input_height = (416, 416)
        in_channels = 32
        x = torch.randn(2, 3, input_height, input_width)
        darknet = Darknet53(in_channels=3)
        outputs = darknet(x)
        self.assertTupleEqual(tuple(outputs['c3'].shape[1:4]), (in_channels * 2**3, input_height / 2**3, input_width / 2**3))
        self.assertTupleEqual(tuple(outputs['c4'].shape[1:4]), (in_channels * 2**4, input_height / 2**4, input_width / 2**4))
        self.assertTupleEqual(tuple(outputs['c5'].shape[1:4]), (in_channels * 2**5 / 2, input_height / 2**5, input_width / 2**5))

    def test_fpn(self):
        input_width, input_height = (416, 416)
        in_channels = 32
        x = torch.randn(2, 3, input_height, input_width)
        darknet = Darknet53(in_channels=3)
        outputs = darknet(x)
        fpn = FPN()
        outputs = fpn(outputs)
        self.assertTupleEqual(tuple(outputs['p3'].shape[1:4]), (in_channels * 2**(3+1), input_height / 2**3, input_width / 2**3))
        self.assertTupleEqual(tuple(outputs['p4'].shape[1:4]), (in_channels * 2**(4+1), input_height / 2**4, input_width / 2**4))
        self.assertTupleEqual(tuple(outputs['p5'].shape[1:4]), (in_channels * 2**5, input_height / 2**5, input_width / 2**5))

    def test_yolo_head(self):
        pass