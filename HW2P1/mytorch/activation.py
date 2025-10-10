import math

import numpy as np
import scipy.special


class Identity:
    def forward(self, Z):
        self.A = Z

        return self.A

    def backward(self, dLdA):
        return dLdA


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))

        return self.A

    def backward(self, dLdA):
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ

        return dLdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """

    def forward(self, Z):
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dLdA):
        dAdZ = 1 - self.A**2
        dLdZ = dLdA * dAdZ
        return dLdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):
        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):
        dZdA = np.where(self.A > 0, 1, 0)
        dLdZ = dZdA * dLdA
        return dLdZ


class GELU:
    """
    Gaussian Error Linear Unit implementation.
    The given autograder does not have unit tests for this activation function,
    so we need to write one ourselves.
    """

    def forward(self, Z):
        self.Z = Z
        self.A = 0.5 * Z * (1 + scipy.special.erf(Z / math.sqrt(2)))

        return self.A

    def backward(self, dLdA):
        dAdZ = 0.5 * (
            1 + scipy.special.erf(self.Z / math.sqrt(2))
        ) + self.Z / math.sqrt(2 * math.pi) * np.exp(-(self.Z**2) / 2)
        dLdZ = dLdA * dAdZ

        return dLdZ


class Softmax:
    """
    Softmax implementation.
    The given autograder does not have unit tests for this activation function,
    so we need to write one ourselves.
    """

    def forward(self, Z):
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exps = np.exp(shift_Z)
        self.A = exps / np.sum(exps, axis=1, keepdims=True)

        return self.A

    def backward(self, dLdA):
        N, C = self.A.shape

        # Initialize an empty list to store each Jacobian
        J = np.zeros((N, C, C))

        # Loop through each sample in the batch
        for i in range(N):
            # Get the activation vector for the i-th sample
            a = self.A[i, :]  # Shape: (C,)

            # Compute the (C, C) Jacobian for this single sample 'a'
            # np.outer computes the off-diagonals: -a_i * a_j
            # np.diag fills the diagonal with: a_i * (1 - a_i)
            J_single = np.diag(a * (1 - a)) - np.outer(a, a)

            J[i, :, :] = J_single

        # Stack the list of 2D arrays into a single 3D tensor
        dLdZ = dLdA * J
        return dLdZ
