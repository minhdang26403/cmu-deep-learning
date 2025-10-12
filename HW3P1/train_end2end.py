import numpy as np
from CTC.CTC import CTCLoss
from models.GRU import GRU
from mytorch.nn.linear import Linear


class GRU_CTC:
    def __init__(self, input_size, hidden_size, num_layers, num_symbols):
        """
        Args:
            input_size: Dimensionality of input features.
            hidden_size: Hidden size of the GRU.
            num_layers: Number of GRU layers.
            num_symbols: Number of output symbols (including blank, assumed index 0).
        """

        # TODO: Initialize the GRU with input_size, hidden_size, and num_layers
        self.gru = GRU(input_size, hidden_size, num_layers)
        # TODO: Define the output size (Hint: What does GRU output per timestep?)
        self.output_size = hidden_size
        # TODO: Initialize a linear layer that maps from output_size to num_symbols
        self.linear = Linear(hidden_size, num_symbols)

    def forward(self, x):
        """
        Forward pass:
            x: shape (seq_len, batch_size, input_size)
        Returns:
            probs: shape (seq_len, batch_size, num_symbols)
        """
        self.gru_out, self.h_n = self.gru.forward(
            x
        )  # (seq_len, batch_size, hidden_size)
        seq_len, batch_size, _ = self.gru_out.shape

        # Flatten the GRU output before feeding it to the Linear Layer.
        x_flat = self.gru_out.reshape(seq_len * batch_size, -1)
        # TODO: Pass through the Linear layer to get logits
        self.logits_flat = self.linear.forward(x_flat)
        # TODO: Reshape logits back to the original shape
        self.logits = self.logits_flat.reshape(seq_len, batch_size, -1)

        # Stabilized softmax:
        # Subtract the max (per example) to avoid overflow, and add a tiny epsilon in the denominator.
        max_logits = np.max(self.logits, axis=2, keepdims=True)
        exp_logits = np.exp(self.logits - max_logits)
        self.probs = exp_logits / (np.sum(exp_logits, axis=2, keepdims=True) + 1e-8)
        return self.probs

    def backward(self, dY):
        """
        Backward pass:
            dY: gradient from CTCLoss, shape (seq_len, batch_size, num_symbols)
        Returns:
            d_x: gradient with respect to the network input,
                 shape (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = dY.shape
        dY_flat = dY.reshape(seq_len * batch_size, -1)

        # TODO: Backprop through Linear layer
        d_gru_out_flat = self.linear.backward(dY_flat)
        d_gru_out = d_gru_out_flat.reshape(seq_len, batch_size, -1)

        # TODO: Backprop through the GRU
        d_x = self.gru.backward(d_gru_out)
        return d_x


def train():
    # --- Hyperparameters ---
    seq_len = 15
    batch_size = 3
    input_size = 8
    hidden_size = 16
    num_layers = 2
    num_symbols = 5
    num_epochs = 10
    learning_rate = 0.001

    # Create the network and CTCLoss.
    model = GRU_CTC(input_size, hidden_size, num_layers, num_symbols)
    ctc_loss_fn = CTCLoss(BLANK=0)

    # --- Create dummy data ---
    # Random input tensor of shape (seq_len, batch_size, input_size)
    x = np.random.randn(seq_len, batch_size, input_size)

    # Create random target sequences (values between 1 and num_symbols-1)
    target_sequences = []
    target_lengths = []
    for b in range(batch_size):
        t_len = np.random.randint(1, seq_len // 2 + 1)
        target_lengths.append(t_len)
        target_seq = np.random.randint(1, num_symbols, size=(t_len,))
        target_sequences.append(target_seq)
    max_target_len = max(target_lengths)
    targets_padded = np.zeros((batch_size, max_target_len), dtype=int)
    for i, seq in enumerate(target_sequences):
        targets_padded[i, : len(seq)] = seq

    # Assume full-length inputs.
    input_lengths = np.full((batch_size,), seq_len, dtype=int)
    target_lengths = np.array(target_lengths, dtype=int)

    # --- Training loop ---
    for epoch in range(num_epochs):
        # TODO: Perform forward pass
        probs = model.forward(x)

        # Calculate the CTCloss
        loss = ctc_loss_fn(probs, targets_padded, input_lengths, target_lengths)
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

        # Compute gradients from CTCLoss.
        dY = ctc_loss_fn.backward()  # shape: (seq_len, batch_size, num_symbols)

        # TODO: Backpropagate through the network
        model.backward(dY)

        # --- Manual parameter update (gradient descent) ---

        # Update parameters in the Linear layer using model.linear.dLdW and model.linear.dLdb
        model.linear.W -= learning_rate * model.linear.dLdW
        model.linear.b -= learning_rate * model.linear.dLdb

        # Update parameters in each GRUCell.

        # Maybe we give them this. We could also not or implement this differently
        for cell in model.gru.layers:
            cell.Wrx -= learning_rate * cell.dWrx
            cell.Wzx -= learning_rate * cell.dWzx
            cell.Wnx -= learning_rate * cell.dWnx
            cell.Wrh -= learning_rate * cell.dWrh
            cell.Wzh -= learning_rate * cell.dWzh
            cell.Wnh -= learning_rate * cell.dWnh
            cell.brx -= learning_rate * cell.dbrx
            cell.bzx -= learning_rate * cell.dbzx
            cell.bnx -= learning_rate * cell.dbnx
            cell.brh -= learning_rate * cell.dbrh
            cell.bzh -= learning_rate * cell.dbzh
            cell.bnh -= learning_rate * cell.dbnh


if __name__ == "__main__":
    train()
