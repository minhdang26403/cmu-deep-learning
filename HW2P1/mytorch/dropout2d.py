# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Dropout2d(object):
    def __init__(self, p=0.5):
        # Dropout probability
        self.p = p
        # The probability of keeping a channel
        self.keep_p = 1 - p

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x, eval=False):
        """
        Arguments:
          x (np.array): (batch_size, in_channel, input_width, input_height)
          eval (boolean): whether the model is in evaluation mode
        Return:
          np.array of same shape as input x
        """
        if eval:
            return x

        # 1. Generate a mask for channels, not pixels.
        # Shape: (batch_size, in_channel). Use the keep probability.
        channel_mask = np.random.binomial(1, self.keep_p, (x.shape[0], x.shape[1]))

        # 2. Expand the mask to the full 4D shape for multiplication.
        # Reshape to (batch, channel, 1, 1) to allow broadcasting.
        self.mask = channel_mask[:, :, np.newaxis, np.newaxis]

        # 3. Apply the mask and scale by the keep probability.
        # Broadcasting automatically tiles the mask across the width and height.
        return x * self.mask / self.keep_p

    def backward(self, delta):
        """
        Arguments:
          delta (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
          np.array of same shape as input delta
        """
        # 1) This method is only called during training.
        # 2) You should scale the result by chain rule
        return delta * self.mask / self.keep_p
