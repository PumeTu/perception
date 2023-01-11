import torch
import torch.nn as nn
from ..layers import BaseConv

class FPN(nn.Module):
    """
    """
    def __init__(self, input: tuple = ('c3', 'c4', 'c5'), output: tuple = ('p3', 'p4', 'p5')):
        super().__init__()
        self.input = input
        self.output = output
        self.conv1 = BaseConv(in_channels=512, out_channels=256, kernel_size=1, stride=1, activation="lrelu")
        self.emb1 = self._make_embedding(512 + 256, 256, 512)
        self.conv2 = BaseConv(in_channels=256, out_channels=128, kernel_size=1, stride=1, activation="lrelu")
        self.emb2 = self._make_embedding(256 + 128, 128, 256)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def _make_embedding(self, in_channels, out_channels, hidden_channels):
        return nn.Sequential(
            *[
                BaseConv(in_channels, out_channels, kernel_size=1, stride=1, activation='lrelu'),
                BaseConv(out_channels, hidden_channels, kernel_size=3, stride=1, activation='lrelu'),
                BaseConv(hidden_channels, out_channels, kernel_size=1, stride=1, activation='lrelu'),
                BaseConv(out_channels, hidden_channels, kernel_size=3, stride=1, activation='lrelu'),
                BaseConv(hidden_channels, out_channels, kernel_size=1, stride=1, activation='lrelu'),
            ]
        )

    def forward(self, x):
        c3, c4, c5 = [x[i] for i in self.input]
        #P4
        m4 = self.conv1(c5)
        m4 = self.upsample(m4)
        m4 = torch.cat([m4, c4], dim=1)
        p4 = self.emb1(m4)
        #P3
        m3 = self.conv2(p4)
        m3 = self.upsample(m3)
        m3 = torch.cat([m3, c3], dim=1)
        p3 = self.emb2(m3)

        return (p3, p4, c5)