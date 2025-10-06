import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = None  # TODO
        self.C = None  # TODO
        se = None  # TODO
        sse = None  # TODO
        mse = None  # TODO

        return NotImplemented

    def backward(self):

        dLdA = None

        return NotImplemented


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = None  # TODO
        C = None  # TODO

        Ones_C = None  # TODO
        Ones_N = None  # TODO

        self.softmax = None  # TODO
        crossentropy = None  # TODO
        sum_crossentropy = None  # TODO
        L = sum_crossentropy / N

        return NotImplemented

    def backward(self):

        dLdA = None  # TODO

        return NotImplemented
