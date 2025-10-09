# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class BatchNorm2d:

    def __init__(self, num_features, alpha=0.9):
        # num features: number of channels
        self.alpha = alpha
        self.eps = 1e-8

        self.Z = None
        self.NZ = None
        self.BZ = None

        self.BW = np.ones((1, num_features, 1, 1))
        self.Bb = np.zeros((1, num_features, 1, 1))
        self.dLdBW = np.zeros((1, num_features, 1, 1))
        self.dLdBb = np.zeros((1, num_features, 1, 1))

        self.M = np.zeros((1, num_features, 1, 1))
        self.V = np.ones((1, num_features, 1, 1))

        # inference parameters
        self.running_M = np.zeros((1, num_features, 1, 1))
        self.running_V = np.ones((1, num_features, 1, 1))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        if eval:
            # TODO
            raise NotImplemented

        self.Z = Z
        self.N = None  # TODO

        self.M = None  # TODO
        self.V = None  # TODO
        self.NZ = None  # TODO
        self.BZ = None  # TODO

        self.running_M = None  # TODO
        self.running_V = None  # TODO

        return self.BZ

    def backward(self, dLdBZ):
        self.dLdBW = None  # TODO
        self.dLdBb = None  # TODO

        dLdNZ = None  # TODO
        dLdV = None  # TODO
        dLdM = None  # TODO

        dLdZ = None  # TODO

        raise NotImplemented
