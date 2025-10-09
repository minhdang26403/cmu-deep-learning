import numpy as np

# Copy your activation.py from HW1P1 here
class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:

    def forward(self, Z):

        self.A = None  # TODO

        return NotImplemented

    def backward(self, dLdA):

        dAdZ = None  # TODO
        dLdZ = None  # TODO

        return NotImplemented


class Tanh:

    def forward(self, Z):

        self.A = None  # TODO

        return NotImplemented

    def backward(self, dLdA):

        dAdZ = None  # TODO
        dLdZ = None  # TODO

        return NotImplemented


class ReLU:

    def forward(self, Z):

        self.A = None  # TODO

        return NotImplemented

    def backward(self, dLdA):

        dAdZ = None  # TODO
        dLdZ = None  # TODO
        
        return NotImplemented
