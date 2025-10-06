import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):

        self.layers = [Linear(2, 3)]
        self.f = [ReLU()]

        self.debug = debug

    def forward(self, A0):

        Z0 = None  # TODO
        A1 = None  # TODO

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return NotImplemented

    def backward(self, dLdA1):

        dA1dZ0 = None  # TODO
        dLdZ0 = None  # TODO
        dLdA0 = None  # TODO

        if self.debug:

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return NotImplemented


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = None  # TODO
        self.f = None  # TODO

        self.debug = debug

    def forward(self, A0):

        Z0 = None  # TODO
        A1 = None  # TODO

        Z1 = None  # TODO
        A2 = None  # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return NotImplemented

    def backward(self, dLdA2):

        dA2dZ1 = None  # TODO
        dLdZ1 = None  # TODO
        dLdA1 = None  # TODO

        dA1dZ0 = None  # TODO
        dLdZ0 = None  # TODO
        dLdA0 = None  # TODO

        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return NotImplemented


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagrmatic view in the writeup for better understanding.
        Use ReLU activation function for all the layers.)
        """
        # List of Hidden Layers
        self.layers = None  # TODO

        # List of Activations
        self.f = None  # TODO

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            Z = None  # TODO
            A = None  # TODO

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return NotImplemented

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = None  # TODO
            dLdZ = None  # TODO
            dLdA = None  # TODO

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return NotImplemented
