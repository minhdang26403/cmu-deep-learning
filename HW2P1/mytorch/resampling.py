import numpy as np


class Upsample1d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1

        Z = np.zeros((batch_size, in_channels, output_width), dtype=A.dtype)
        Z[:, :, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        dLdA = dLdZ[:, :, :: self.upsampling_factor]

        return dLdA


class Downsample1d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        _, _, input_width = A.shape
        self.input_width = input_width
        Z = A[:, :, :: self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, _ = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_width), dtype=dLdZ.dtype)
        dLdA[:, :, :: self.downsampling_factor] = dLdZ

        return dLdA


class Upsample2d:
    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size, in_channels, input_height, input_width = A.shape
        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1
        Z = np.zeros(
            (batch_size, in_channels, output_height, output_width), dtype=A.dtype
        )
        Z[:, :, :: self.upsampling_factor, :: self.upsampling_factor] = A

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        dLdA = dLdZ[:, :, :: self.upsampling_factor, :: self.upsampling_factor]

        return dLdA


class Downsample2d:
    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        _, _, self.input_height, self.input_width = A.shape
        Z = A[:, :, :: self.downsampling_factor, :: self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, _, _ = dLdZ.shape
        dLdA = np.zeros(
            (batch_size, in_channels, self.input_height, self.input_width),
            dtype=dLdZ.dtype,
        )
        dLdA[:, :, :: self.downsampling_factor, :: self.downsampling_factor] = dLdZ

        return dLdA
