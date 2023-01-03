import unittest
import torch
from ..perception.model.model import Darknet53, FPN

class TestModel(unittest.TestCase):
    def test_darknet53(self):
        input_width, input_height = (416, 416)
        x = torch.randn(2, 3, input_height, input_width)
        darknet = Darknet53(in_channels=3)
        c3, c4, c5 = darknet(x)
        assert c3.shape[2, 3] == (input_height / 2**3, input_width / 2**3)

