import numpy as np


class CTC(object):
    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        skip_connect = np.zeros(N)
        for i in range(len(target)):
            if i > 0 and target[i] != target[i - 1]:
                skip_connect[2 * i + 1] = 1
        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and
        # 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # 1. Correct Initialization
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        # 2. Main Recursive Loop
        for t in range(1, T):
            for s in range(S):
                # Get the network probability for the current character
                char_idx = extended_symbols[s]
                prob = logits[t][char_idx]

                # Get the sum of probabilities from previous states at t-1
                # Path 1: From the same state s
                recursive_sum = alpha[t - 1][s]

                # Path 2: From the previous state s-1
                if s > 0:
                    recursive_sum += alpha[t - 1][s - 1]

                # Path 3: The "skip connection" from state s-2
                # This is only possible for non-blank symbols where skip_connect is True
                if (
                    s > 1 and extended_symbols[s] != 0 and skip_connect[s]
                ):  # Assuming blank is index 0
                    recursive_sum += alpha[t - 1][s - 2]

                # Update alpha for the current state and time
                alpha[t][s] = recursive_sum * prob

        # <---------------------------------------------

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # 1. Correct Initialization at the last timestep (T-1)
        beta[T - 1][S - 1] = 1
        beta[T - 1][S - 2] = 1

        # 2. Main Recursive Loop (iterating backwards from T-2 to 0)
        for t in range(T - 2, -1, -1):
            for s in range(S - 1, -1, -1):
                # Get the sum of probabilities of paths starting from state s at time t
                recursive_sum = 0.0

                # Path 1: Transition from s at t to s at t+1
                char_idx_s = extended_symbols[s]
                recursive_sum += beta[t + 1][s] * logits[t + 1][char_idx_s]

                # Path 2: Transition from s at t to s+1 at t+1
                if s < S - 1:
                    char_idx_s_plus_1 = extended_symbols[s + 1]
                    recursive_sum += (
                        beta[t + 1][s + 1] * logits[t + 1][char_idx_s_plus_1]
                    )

                # Path 3: Skip transition from s at t to s+2 at t+1
                # A skip FROM s is valid if the destination s+2 allows it
                if s < S - 2 and skip_connect[s + 2]:
                    char_idx_s_plus_2 = extended_symbols[s + 2]
                    recursive_sum += (
                        beta[t + 1][s + 2] * logits[t + 1][char_idx_s_plus_2]
                    )

                beta[t][s] = recursive_sum

        # <--------------------------------------------

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # We can compute the total probability of the sequence from the final alphas
        # This will be our normalization constant.
        total_prob = alpha[-1, -1] + alpha[-1, -2]

        # Iterate through each timestep
        for t in range(T):
            # Calculate the product of alpha and beta for all states at the current time
            # step
            gamma[t, :] = alpha[t, :] * beta[t, :]

            # Normalize by the total probability of the sequence
            # Add a small epsilon to avoid division by zero
            gamma[t, :] /= total_prob + 1e-9

        # <---------------------------------------------

        return gamma


class CTCLoss(object):
    def __init__(self, BLANK=0):
        """

                Initialize instance variables

        Argument(s)
                -----------
                BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            ctc = CTC()
            target_trunc = target[batch_itr, : target_lengths[batch_itr]]
            logits_trunc = logits[: input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = ctc.extend_target_with_blank(target_trunc)
            alpha = ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)
            beta = ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)
            gamma = ctc.get_posterior_probs(alpha, beta)

            div = 0
            S, T = len(extended_symbols), len(logits_trunc)
            for t in range(T):
                for s in range(S):
                    div -= gamma[t, s] * np.log(logits_trunc[t, extended_symbols[s]])

            total_loss[batch_itr] = div

            # <---------------------------------------------

        total_loss = np.mean(total_loss)

        return total_loss

    def backward(self):
        """

                CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative
                w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            ctc = CTC()
            target_trunc = self.target[batch_itr, : self.target_lengths[batch_itr]]
            logits_trunc = self.logits[: self.input_lengths[batch_itr], batch_itr, :]
            extended_symbols, skip_connect = ctc.extend_target_with_blank(target_trunc)
            alpha = ctc.get_forward_probs(logits_trunc, extended_symbols, skip_connect)
            beta = ctc.get_backward_probs(logits_trunc, extended_symbols, skip_connect)
            gamma = ctc.get_posterior_probs(alpha, beta)

            S, T = len(extended_symbols), len(logits_trunc)
            for t in range(T):
                for s in range(S):
                    dY[t, batch_itr, extended_symbols[s]] -= (
                        gamma[t, s] / logits_trunc[t, extended_symbols[s]]
                    )

            # <---------------------------------------------

        return dY
