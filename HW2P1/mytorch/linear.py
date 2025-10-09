# Copy your Linear class from HW1P1 here

import numpy as np

class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = None  # TODO
        self.b = None  # TODO

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = None  # TODO
        self.N = None  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        # self.Ones = np.ones((self.N,1))
        Z = None  # TODO

        return NotImplemented

    def backward(self, dLdZ):

        dLdA = None  # TODO
        self.dLdW = None  # TODO
        self.dLdb = None  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return NotImplemented
