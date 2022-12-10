import torch
import torch.nn as nn

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported activation function: {}".format(name))
    return module

class BaseConv(nn.Module):
    '''
    Basic Convloutional Block (Conv -> BatchNorm -> ReLU/SiLU)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of kernel
        stride (int): stride length
        groups (int): convolution group
        bias (bool): add bias or not
        activation (str): name of activation function
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, activation='silu'):
        padding = (kernel_size - 1) // 2 #same padding
        self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation, inplace=True)

    def forward(self, x):
        return self.act(self.batchnorm(self.conv(x)))

class Residual(nn.Module):
    '''
    Basic Residual Block as defined in the ResNet paper with two convolutional layers and a skip connection
    Args:
        in_channels (int): number of input channels
    '''
    def __init__(self, in_channels):
        reduced_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, reduced_channels, kernel_size=1, stride=1, activation='lrelu')
        self.conv2 = BaseConv(reduced_channels, in_channels, kernel_size=3, stride=1, activation='lrelu')

    def forward(self, x):
        out = self.conv2(self.conv2(x))
        return x + out