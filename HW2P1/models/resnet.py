import sys

sys.path.append("mytorch")

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layers = []  # TODO

    def forward(self, A):
        # TODO
        return NotImplemented

    def backward(self, grad):
        # TODO
        return NotImplemented


class ResBlock(object):

    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
        # TODO
        self.convolution_layers = []
        self.final_activation = None
        # TODO
        if stride != 1 or in_channels != out_channels or filter_size != 1 or padding != 0:
            self.residual_connection = None
        else:
            self.residual_connection = None

    def forward(self, A):
        # TODO
        return NotImplemented

    def backward(self, grad):
        # TODO
        return NotImplementedError
