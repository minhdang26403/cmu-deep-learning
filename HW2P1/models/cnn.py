# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

import sys

import numpy as np
from Conv1d import Conv1d
from flatten import Flatten
from linear import Linear

sys.path.append("mytorch")


class CNN(object):
    """
    A simple convolutional neural network
    """

    def __init__(
        self,
        input_width,
        num_input_channels,
        num_channels,
        kernel_sizes,
        strides,
        num_linear_neurons,
        activations,
        conv_weight_init_fn,
        bias_init_fn,
        linear_weight_init_fn,
        criterion,
        lr,
    ):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)

        self.convolutional_layers = [
            Conv1d(
                in_channels=num_input_channels,
                out_channels=num_channels[0],
                kernel_size=kernel_sizes[0],
                stride=strides[0],
            ),
            Conv1d(
                in_channels=num_channels[0],
                out_channels=num_channels[1],
                kernel_size=kernel_sizes[1],
                stride=strides[1],
            ),
            Conv1d(
                in_channels=num_channels[1],
                out_channels=num_channels[2],
                kernel_size=kernel_sizes[2],
                stride=strides[2],
            ),
        ]  # TODO
        self.flatten = Flatten()  # TODO
        conv1_out = (input_width - kernel_sizes[0]) / strides[0] + 1
        conv2_out = (conv1_out - kernel_sizes[1]) / strides[1] + 1
        conv3_out = (conv2_out - kernel_sizes[2]) / strides[2] + 1
        self.linear_layer = Linear(
            int(conv3_out * num_channels[2]), num_linear_neurons
        )  # TODO

        self.layers = [
            self.convolutional_layers[0],
            activations[0],
            self.convolutional_layers[1],
            activations[1],
            self.convolutional_layers[2],
            activations[2],
            self.flatten,
            self.linear_layer,
        ]
        # <---------------------

        # Don't change this -->
        out_features, in_features = self.linear_layer.W.shape
        if linear_weight_init_fn is not None:
            self.linear_layer.W = linear_weight_init_fn(out_features, in_features)
        if bias_init_fn is not None:
            self.linear_layer.b = bias_init_fn(out_features)
        # <---------------------

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Your code goes here -->
        for layer in self.layers:
            A = layer.forward(A)
        # Iterate through each layer
        # <---------------------

        # Save output (necessary for error and loss)
        self.Z = A

        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Your code goes here -->
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)
        # Iterate through each layer in reverse order
        # <---------------------

        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW = np.zeros(self.linear_layer.W.shape)
        self.linear_layer.dLdb = np.zeros(self.linear_layer.b.shape)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (
                self.convolutional_layers[i].conv1d_stride1.W
                - self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW
            )
            self.convolutional_layers[i].conv1d_stride1.b = (
                self.convolutional_layers[i].conv1d_stride1.b
                - self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb
            )

        self.linear_layer.W = self.linear_layer.W - self.lr * self.linear_layer.dLdW
        self.linear_layer.b = self.linear_layer.b - self.lr * self.linear_layer.dLdb

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
