import numpy as np
from resampling import Downsample2d


class Conv2d_stride1:
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, _, input_height, input_width = A.shape
        output_height = input_height - self.kernel_size + 1
        output_width = input_width - self.kernel_size + 1
        Z = np.zeros(
            (batch_size, self.out_channels, output_height, output_width),
            dtype=self.A.dtype,
        )
        for i in range(output_height):
            for j in range(output_width):
                Z[:, :, i, j] = (
                    np.tensordot(
                        A[:, :, i : i + self.kernel_size, j : j + self.kernel_size],
                        self.W,
                        axes=([1, 2, 3], [1, 2, 3]),
                    )
                    + self.b
                )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        _, _, output_height, output_width = dLdZ.shape
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dLdW[:, :, i, j] = np.tensordot(
                    dLdZ,
                    self.A[:, :, i : i + output_height, j : j + output_width],
                    axes=([0, 2, 3], [0, 2, 3]),
                )
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        # 1. Flip the kernel the filter top to bottom and left to right.
        W_flipped = np.flip(self.W, axis=(2, 3))

        # 2. Pad dLdZ with 0s with (kernel_size - 1) zeroes on both sides
        padding_amount = self.kernel_size - 1
        pad_width = (
            (0, 0),
            (0, 0),
            (padding_amount, padding_amount),
            (padding_amount, padding_amount),
        )
        dLdZ_padded = np.pad(
            dLdZ, pad_width=pad_width, mode="constant", constant_values=0
        )

        # 3. Convolve to find dLdA
        dLdA = np.zeros(self.A.shape)
        _, _, input_height, input_width = self.A.shape

        for i in range(input_height):
            for j in range(input_width):
                dLdA[:, :, i, j] = np.tensordot(
                    dLdZ_padded[
                        :, :, i : i + self.kernel_size, j : j + self.kernel_size
                    ],
                    W_flipped,
                    axes=([1, 2, 3], [0, 2, 3]),
                )

        return dLdA


class Conv2d:
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

        # Initialize Conv2d_stride1() and Downsample2d() instance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        padding_amount = self.padding
        pad_width = (
            (0, 0),
            (0, 0),
            (padding_amount, padding_amount),
            (padding_amount, padding_amount),
        )
        A_padded = np.pad(A, pad_width=pad_width, mode="constant", constant_values=0)

        # Call conv2d_stride1
        Z = self.conv2d_stride1.forward(A_padded)

        # Call downsample2d
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        dLdA = dLdA[:, :, self.padding : -self.padding, self.padding : -self.padding]

        return dLdA
