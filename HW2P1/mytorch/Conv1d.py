# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A

        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        self.dLdW = None  # TODO
        self.dLdb = None  # TODO
        dLdA = None  # TODO

        return NotImplemented


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.padding = padding

        # Initialize Conv1d_stride1() and Downsample1d() instance
        self.conv1d_stride1 = None  # TODO
        self.downsample1d = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # Pad the input appropriately using np.pad() function
        # TODO

        # Call conv1d_stride1
        # TODO

        # Call downsample1d
        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO

        # Call conv1d_stride1 backward
        dLdA = None  # TODO
        
        # Unpad the gradient
        # TODO

        return NotImplemented
