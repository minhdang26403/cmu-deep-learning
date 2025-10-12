import numpy as np
from mytorch.gru_cell import GRUCell


class GRU:
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Build a list of GRUCell instances—one per layer.
        self.layers = []

        for i in range(num_layers):
            # TODO : input_size if i == 0 else hidden_size
            input_shape = input_size if i == 0 else hidden_size
            # TODO : what should be passed to the second argument?
            gru_cell = GRUCell(input_shape, hidden_size)
            self.layers.append(gru_cell)

        self.caches = [[] for _ in range(num_layers)]

    def forward(self, input_seq, h0=None):
        """
        Args:
            input_seq: np.array of shape (seq_len, batch_size, input_size)
            h0: (optional) np.array of shape (num_layers, batch_size, hidden_size);
                if None, zeros are used.
        Returns:
            out: np.array of shape (seq_len, batch_size, hidden_size)
            h_n: final hidden states, shape (num_layers, batch_size, hidden_size)
        """
        seq_len, batch_size, _ = input_seq.shape

        # TODO If h0 is None then it should be set to a zeros np array of appropriate size
        if h0 is None:
            h0 = np.zeros((self.num_layers, batch_size, self.hidden_size))

        assert h0 is not None, "h0 needs to be zeros "

        out = input_seq
        self.hiddens = []  # store list of hidden states per layer
        for l_i, cell in enumerate(self.layers):
            layer_cache = []  # will hold one cache per time step
            layer_output = []  # will store the output at each time step
            h0_i = l_i  # TODO : refer the appropriate index from h0
            h_prev = h0[h0_i]
            for t in range(seq_len):
                x_t = out[t]  # shape: (batch_size, input_dim)
                # TODO : add zeros of size (batch_size, hidden_size)
                new_h = np.zeros((batch_size, self.hidden_size))
                cache_t = []  # one cache per sample in the batch at time t

                # take the current layer throughout
                for b in range(batch_size):
                    # Compute cell forward for sample b
                    # TODO think about what you will use for the x and prev h here
                    h_new = cell.forward(x_t[b], h_prev[b])
                    new_h[b] = h_new

                    # TODO save a copy of the cell’s internal values. (recall the state variables from GRUCell?)
                    # hint: use .copy() to create the copy
                    cache_t.append(
                        {
                            "x": cell.x.copy(),
                            "hidden": cell.hidden.copy(),
                            "r": cell.r.copy(),
                            "z": cell.z.copy(),
                            "n": cell.n.copy(),
                        }
                    )

                layer_cache.append(cache_t)
                layer_output.append(new_h)
                h_prev = new_h  # TODO : not that new_h
            self.caches[l_i] = layer_cache
            self.hiddens.append(new_h)  # final hidden state for this layer

            # Prepare input for next layer
            out = np.stack(
                layer_output, axis=0
            )  # now shape (seq_len, batch_size, hidden_size)

        h_n = np.stack(
            self.hiddens, axis=0
        )  # shape: (num_layers, batch_size, hidden_size)
        return out, h_n

    def backward(self, d_out):
        """
        Backward pass through the GRU.
        Args:
            d_out: gradient with respect to the GRU output,
                   shape (seq_len, batch_size, hidden_size)
        Returns:
            d_input: gradient with respect to the GRU input,
                     shape (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = d_out.shape
        d_layer = d_out  # gradient flowing into the last layer

        # Loop over layers in reverse order.
        for l_i in reversed(range(self.num_layers)):
            cell = self.layers[l_i]
            layer_cache = self.caches[l_i]
            # TODO zeros array of size batch_size, self.hidden_size))
            d_h_next = np.zeros((batch_size, self.hidden_size))
            # TODO np.zeros((seq_len, batch_size, cell.Wrx.shape[1]))
            d_input_layer = np.zeros((seq_len, batch_size, cell.Wrx.shape[1]))
            # Loop backward in time.
            for t in reversed(range(seq_len)):
                # Total gradient at time t is gradient from above plus gradient flowing from later time.
                d_h = d_layer[t] + d_h_next  # shape: (batch_size, hidden_size)

                # TODO : make a zeros array (think about what will be the shape)
                d_x_t = np.zeros((batch_size, cell.Wrx.shape[1]))

                # TODO : make a zeros array (think about what will be the shape)
                d_h_prev = np.zeros((batch_size, self.hidden_size))

                for b in range(batch_size):
                    # Restore the cell's internal variables.
                    # TODO : read appropriate values from the cache
                    cache = layer_cache[t][b]
                    cell.x, cell.hidden, cell.r, cell.z, cell.n = (
                        cache["x"],
                        cache["hidden"],
                        cache["r"],
                        cache["z"],
                        cache["n"],
                    )

                    # Call the cell's backward.
                    dx, dh_prev = cell.backward(
                        d_h[b]
                    )  # TODO : send the right batch from d_h
                    d_x_t[b] = (
                        dx  # TODO : think of the appropriate value to be used here
                    )
                    d_h_prev[b] = (
                        dh_prev  # TODO : think of the appropriate value to be used here
                    )

                d_input_layer[t] = d_x_t  # pass gradient for input time step
                d_h_next = d_h_prev  # pass gradient to previous time step
            d_layer = d_input_layer  # TODO : think about what needs to be propagated backwards from the GRU layers
        return d_layer  # gradient with respect to the GRU input
