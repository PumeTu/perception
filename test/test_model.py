import unittest
import torch
from perception.models.model import Darknet53

class TestModel(unittest.TestCase):
    def test_darknet53(self):
        input_width, input_height = (416, 416)
        x = torch.randn(2, 3, input_height, input_width)
        darknet = Darknet53(in_channels=3)
        outputs = darknet(x)
        self.assertTupleEqual(tuple(outputs['c3'].shape[2:4]), ((input_height / 2**3), (input_width / 2**3)))
        self.assertTupleEqual(tuple(outputs['c4'].shape[2:4]), ((input_height / 2**4), (input_width / 2**4)))
        self.assertTupleEqual(tuple(outputs['c5'].shape[2:4]), ((input_height / 2**5), (input_width / 2**5)))

    def test_spp(self):
        pass

    def test_fpn(self):
        pass

    def test_yolo_head(self):
        pass