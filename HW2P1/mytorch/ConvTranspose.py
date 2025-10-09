import numpy as np
from resampling import *
from Conv1d import *
from Conv2d import *


class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.upsampling_factor = upsampling_factor

        # Initialize Upsample1d() and Conv1d_stride1() instance
        self.upsample1d = None  # TODO
        self.conv1d_stride1 = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # TODO
        # Call upsample1d
        A_upsampled = None  # TODO

        # Call conv1d_stride1()
        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        # Call backward in the correct order
        delta_out = None  # TODO

        dLdA = None  # TODO

        return NotImplemented


class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.upsampling_factor = upsampling_factor

        # Initialize Upsample2d() and Conv2d_stride1() instance
        self.upsample2d = None  # TODO
        self.conv2d_stride1 = None  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample2d
        A_upsampled = None  # TODO

        # Call conv2d_stride1()
        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call backward in correct order
        delta_out = None  # TODO

        dLdA = None  # TODO

        return dLdA
