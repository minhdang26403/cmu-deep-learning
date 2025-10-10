import sys

sys.path.append("mytorch")

from activation import *
from batchnorm2d import *
from Conv2d import *


class ConvBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.layers = [
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels),
        ]  # TODO

    def forward(self, A):
        # TODO
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, grad):
        # TODO
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class ResBlock(object):
    def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
        # TODO
        self.convolution_layers = [
            ConvBlock(in_channels, out_channels, filter_size, stride, padding),
            ReLU(),
            ConvBlock(out_channels, out_channels, 1, 1, 0),
        ]
        self.final_activation = ReLU()
        # TODO
        if (
            stride != 1
            or in_channels != out_channels
            or filter_size != 1
            or padding != 0
        ):
            self.residual_connection = ConvBlock(
                in_channels, out_channels, filter_size, stride, padding
            )
        else:
            self.residual_connection = Identity()

    def forward(self, A):
        # TODO
        residual = A

        # Main path
        conv_out = A
        for layer in self.convolution_layers:
            conv_out = layer.forward(conv_out)

        # Residual path
        residual_out = self.residual_connection.forward(residual)

        # Add the two paths
        return self.final_activation.forward(conv_out + residual_out)

    def backward(self, grad):
        # TODO
        # Backprop through the final activation
        grad = self.final_activation.backward(grad)

        # Gradients to both paths are the same
        grad_for_conv = grad
        grad_for_residual = grad

        # Backprop through the residual path
        grad_from_residual = self.residual_connection.backward(grad_for_residual)

        # Backprop through the main path IN REVERSE ORDER
        grad_from_conv = grad_for_conv
        for layer in reversed(self.convolution_layers):
            grad_from_conv = layer.backward(grad_from_conv)

        # Total gradient is the sum from both paths
        return grad_from_conv + grad_from_residual
