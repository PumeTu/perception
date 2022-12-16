import torch 
import torch.nn as nn

def get_activation(name: str ="silu", inplace: bool =True):
    '''
    Get an activation function given the name
    Args:
        name (str): name of desired activation function
        inplace (bool): specify whether to the operation inplace or not
    Returns
        module (nn.Module): activation function requested
    '''
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, groups:int =1, bias: bool =False, activation: str ='silu'):
        super().__init__()
        padding = (kernel_size - 1) // 2 #same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(activation, inplace=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.act(self.batchnorm(self.conv(x)))

class Residual(nn.Module):
    '''
    Basic Residual Block as defined in the ResNet paper with two convolutional layers and a skip connection
    Args:
        in_channels (int): number of input channels
    '''
    def __init__(self, in_channels: int):
        super().__init__()
        reduced_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, reduced_channels, kernel_size=1, stride=1, activation='lrelu')
        self.conv2 = BaseConv(reduced_channels, in_channels, kernel_size=3, stride=1, activation='lrelu')

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.conv2(self.conv1(x))
        return x + out

class SPPBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple = (5, 9, 13), activation='silu'):
        super().__init__()
        reduced_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, reduced_channels, kernel_size=1, stride=1, activation=activation)
        self.maxpool = nn.Module([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks//2) for ks in kernel_sizes])
        expanded_channels = reduced_channels * (len(kernel_sizes) + 1)        
        self.conv2 = BaseConv(expanded_channels, out_channels, kernel_size=1, stride=1, activation=activation)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)   
        return x
        
        