# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import Downsample1d


class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
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
        batch_size, _, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size), dtype=self.A.dtype)

        for i in range(output_size):
            Z[:, :, i] = (
                np.tensordot(
                    self.A[:, :, i : i + self.kernel_size],
                    self.W,
                    axes=([1, 2], [1, 2]),
                )
                + self.b
            )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        _, _, output_size = dLdZ.shape
        for k in range(self.kernel_size):
            self.dLdW[:, :, k] = np.tensordot(
                dLdZ,
                self.A[:, :, k : k + output_size],
                axes=([0, 2], [0, 2]),
            )

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        # --- Calculate dLdA ---

        # 1. Flip the kernel left-to-right once before the loop
        W_flipped = np.flip(self.W, axis=2)

        # 2. Pad dLdZ once before the loop
        padding_amount = self.kernel_size - 1
        pad_width = ((0, 0), (0, 0), (padding_amount, padding_amount))
        dLdZ_padded = np.pad(
            dLdZ, pad_width=pad_width, mode="constant", constant_values=0
        )

        # 3. Convolve to find dLdA
        _, _, input_size = self.A.shape
        dLdA = np.zeros(self.A.shape)

        for i in range(input_size):
            dLdA[:, :, i] = np.tensordot(
                dLdZ_padded[:, :, i : i + self.kernel_size],
                W_flipped,
                axes=([1, 2], [0, 2]),
            )

        return dLdA


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.padding = padding

        # Initialize Conv1d_stride1() and Downsample1d() instance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        padding_amount = self.padding
        pad_width = ((0, 0), (0, 0), (padding_amount, padding_amount))
        A_padded = np.pad(A, pad_width=pad_width, mode="constant", constant_values=0)

        # Call conv1d_stride1
        Z = self.conv1d_stride1.forward(A_padded)

        # Call downsample1d
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.padding : -self.padding]

        return dLdA
