import numpy as np
from resampling import Downsample2d


class MaxPool2d_stride1:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel_size + 1
        output_height = input_height - self.kernel_size + 1

        Z = np.zeros(
            (batch_size, in_channels, output_width, output_height),
            dtype=self.A.dtype,
        )
        # Cache to store the (row, col) coordinates of the max value for each patch
        self.max_indices = np.zeros_like(Z, dtype=object)

        for i in range(output_width):
            for j in range(output_height):
                # Isolate the current patch
                patch = A[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                # Find the max value in the patch
                Z[:, :, i, j] = np.max(patch, axis=(2, 3))

                # Find the index of the max in the patch and store its global
                # coordinates.
                for n in range(batch_size):
                    for c in range(in_channels):
                        patch_2d = patch[n, c]
                        # Find the 2D index of the max within the 2D patch
                        max_idx_local = np.unravel_index(
                            np.argmax(patch_2d), patch_2d.shape
                        )
                        # Convert to global coordinates and store
                        self.max_indices[n, c, i, j] = (
                            i + max_idx_local[0],
                            j + max_idx_local[1],
                        )

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros_like(self.A)

        batch_size, in_channels, output_height, output_width = dLdZ.shape

        # Loop through each element of the output/incoming gradient
        for n in range(batch_size):
            for c in range(in_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        # Get the coordinates of the "winner" from the forward pass
                        (max_row, max_col) = self.max_indices[n, c, i, j]

                        # Add the incoming gradient to that specific "winner" location
                        # in dLdA. We use += because one input might be the max for
                        # multiple patches
                        dLdA[n, c, max_row, max_col] += dLdZ[n, c, i, j]

        return dLdA


class MeanPool2d_stride1:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel_size + 1
        output_height = input_height - self.kernel_size + 1
        Z = np.zeros(
            (batch_size, in_channels, output_width, output_height),
            dtype=self.A.dtype,
        )

        for i in range(output_width):
            for j in range(output_height):
                patch = A[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                Z[:, :, i, j] = np.mean(patch, axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        # 1. Pad the incoming gradient
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

        _, _, input_height, input_width = self.A.shape
        dLdA = np.zeros_like(self.A)

        # 2. Create the simple 2D distribution kernel
        # This kernel is the same for every channel.
        num_elements = self.kernel_size * self.kernel_size
        W_2D = np.ones((self.kernel_size, self.kernel_size)) / num_elements

        for i in range(input_height):
            for j in range(input_width):
                # Get the slice of the padded gradient
                dLdZ_slice = dLdZ_padded[
                    :, :, i : i + self.kernel_size, j : j + self.kernel_size
                ]

                # Use broadcasting to multiply the 2D kernel with each channel of the 4D
                # slice and sum over the kernel dimensions (the last two axes).
                dLdA[:, :, i, j] = np.sum(dLdZ_slice * W_2D, axis=(2, 3))

        return dLdA


class MaxPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)
        return dLdA


class MeanPool2d:
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)
        return dLdA
