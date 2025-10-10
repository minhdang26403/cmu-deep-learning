import activation
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
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (self.A - self.Y) * (self.A - self.Y)  # TODO

        # Ones_C = np.ones(self.C, dtype="f")
        # Ones_N = np.ones(self.N, dtype="f")

        # sse = Ones_N.T @ se @ Ones_C  # TODO
        sse = np.sum(se)
        mse = sse / (2 * self.N * self.C)  # TODO

        return mse

    def backward(self):
        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


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
        self.N = self.A.shape[0]  # TODO
        self.C = self.A.shape[1]  # TODO

        # Ones_C = np.ones(self.C, dtype="f")  # TODO
        # Ones_N = np.ones(self.N, dtype="f")  # TODO

        self.softmax = activation.Softmax().forward(self.A)  # TODO

        # crossentropy = -self.Y * np.log(self.softmax) @ Ones_C  # TODO
        # sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        crossentropy = np.sum(-self.Y * np.log(self.softmax), axis=1)  # TODO
        sum_crossentropy = np.sum(crossentropy, axis=0)  # TODO

        L = sum_crossentropy / self.N

        return L

    def backward(self):
        dLdA = self.softmax - self.Y  # TODO

        return dLdA
