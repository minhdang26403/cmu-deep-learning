import numpy as np


class Dropout:
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.seed = seed
        self.mask = None

        if seed is not None:
            np.random.seed(seed)

    def forward(self, A, training=True):
        if training:
            self.mask = np.random.binomial(1, self.p, size=A.shape).astype(A.dtype)
            A_drop = (self.mask * A) / self.p
        else:
            A_drop = A

        return A_drop

    def backward(self, dLdA, training=True):
        if not training:
            return dLdA

        return (self.mask * dLdA) / self.p
