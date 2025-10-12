import numpy as np


class GreedySearchDecoder(object):
    def __init__(self, symbol_set):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        _, seq_len, batch_size = y_probs.shape
        for batch_itr in range(batch_size):
            path = ""
            for t in range(seq_len):
                logits = y_probs[:, t, batch_itr]
                max_prob = np.max(logits, axis=0)
                path_prob *= max_prob
                max_idx = np.argmax(logits, axis=0)
                # Skip blank (index 0) and repeated symbols
                if max_idx != blank and (
                    len(path) == 0 or path[-1] != self.symbol_set[max_idx - 1]
                ):
                    path += self.symbol_set[max_idx - 1]
            decoded_path.append(path)

        return decoded_path[0], path_prob


class BeamSearchDecoder(object):
    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """

        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
                        batch size for part 1 will remain 1, but if you plan to use your
                        implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]

        best_paths = {" ": 1}
        temp_best_paths = {}

        for t in range(T):
            temp_best_paths = {}
            logits = y_probs[:, t]
            for path, score in best_paths.items():
                for s, prob in enumerate(logits):
                    sym = self.symbol_set[s - 1] if s > 0 else " "
                    new_path = path

                    if path[-1] == " ":
                        new_path = new_path[:-1] + sym
                    elif (path[-1] != sym) and not (t == T - 1 and sym == " "):
                        new_path += sym

                    if new_path in temp_best_paths:
                        temp_best_paths[new_path] += prob * score
                    else:
                        temp_best_paths[new_path] = prob * score

            if len(temp_best_paths) >= self.beam_width:
                best_paths = dict(
                    sorted(temp_best_paths.items(), key=lambda x: x[1], reverse=True)[
                        : self.beam_width
                    ]
                )

        best_path = max(best_paths, key=best_paths.get)
        merged_path_scores = {}
        for path, score in temp_best_paths.items():
            path = path.replace(" ", "")
            merged_path_scores[path] = score

        return best_path, merged_path_scores
