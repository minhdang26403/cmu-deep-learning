import os
import sys

# Add the parent directory (i.e. hw3) to sys.path so that modules in models, CTC, mytorch, etc. can be found.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

import importlib
import io
import re
from contextlib import redirect_stdout

import numpy as np

# =============================================================================
# Import the student's GRU class from models/GRU.py using importlib.
# =============================================================================
try:
    studentGRUModule = importlib.import_module("models.GRU")
    if hasattr(studentGRUModule, "GRU"):
        GRU = getattr(studentGRUModule, "GRU")
        if not callable(GRU):
            raise TypeError("The imported GRU is not callable (it might be a module).")
    else:
        raise AttributeError("Module 'models.GRU' has no attribute 'GRU'.")
except Exception as e:
    print("DEBUG INFORMATION:")
    print(f"Error importing student's GRU: {e}")
    print("FINAL SCORES:")
    print("GRU: 0 / 10")
    print("train_end2end: 0 / 10")
    sys.exit(1)


# =============================================================================
# Helper functions
# =============================================================================
def load_solution_data(filename="autograder/solution_outputs.npz"):
    if not os.path.exists(filename):
        return None, f"Solution output file '{filename}' not found."
    data = np.load(filename)
    return data, None


def compute_error_metrics(student, solution):
    abs_diff = np.abs(student - solution)
    mean_diff = np.mean(abs_diff)
    mean_solution = np.mean(np.abs(solution))
    relative_error = mean_diff / (mean_solution + 1e-8)
    max_diff = np.max(abs_diff)
    return mean_diff, relative_error, max_diff


# =============================================================================
# Test functions for each problem
# =============================================================================
def test_gru():
    """
    Runs tests for the GRU implementation:
      - Forward pass test (5 points)
      - Backward pass test (5 points)
    Returns a dictionary with a total score (max 10 points) and detailed feedback.
    """
    debug_lines = []
    score = 0
    max_score = 10

    # Fixed parameters and seed for reproducibility.
    np.random.seed(11785)
    seq_len = 7
    batch_size = 4
    input_size = 10
    hidden_size = 20
    num_layers = 2

    # Generate input and initial hidden state.
    x = np.random.randn(seq_len, batch_size, input_size)
    h0 = np.zeros((num_layers, batch_size, hidden_size))

    # Instantiate student's GRU.
    try:
        gru = GRU(input_size, hidden_size, num_layers=num_layers)
        debug_lines.append("Successfully instantiated student's GRU.")
    except Exception as e:
        debug_lines.append(f"Error instantiating student's GRU: {e}")
        return {"score": 0, "max_score": max_score, "feedback": "\n".join(debug_lines)}

    # ------------------------------
    # Forward Pass Test (5 points)
    # ------------------------------
    try:
        student_out, student_h_n = gru.forward(x, h0)
        debug_lines.append("Forward pass executed successfully.")
    except Exception as e:
        debug_lines.append(f"Error during forward pass: {e}")
        return {"score": 0, "max_score": max_score, "feedback": "\n".join(debug_lines)}

    data, err = load_solution_data()
    if err:
        debug_lines.append(err)
        return {"score": 0, "max_score": max_score, "feedback": "\n".join(debug_lines)}

    solution_out = data["out"]
    solution_h_n = data["h_n"]

    forward_close = np.allclose(student_out, solution_out, rtol=1e-5, atol=1e-7)
    hidden_close = np.allclose(student_h_n, solution_h_n, rtol=1e-5, atol=1e-7)

    if forward_close and hidden_close:
        score += 5  # full points for forward pass.
        debug_lines.append("Forward pass outputs match expected outputs.")
    else:
        mean_diff, rel_err, max_diff = compute_error_metrics(student_out, solution_out)
        debug_lines.append(
            f"Forward pass outputs differ: Mean diff: {mean_diff:.3e}, relative error: {rel_err:.3e}, max diff: {max_diff:.3e}."
        )

    # ------------------------------
    # Backward Pass Test (5 points)
    # ------------------------------
    try:
        d_out = np.random.randn(*student_out.shape)
        student_d_input = gru.backward(d_out)
        debug_lines.append("Backward pass executed successfully.")
    except Exception as e:
        debug_lines.append(f"Error during backward pass: {e}")
        return {
            "score": score,
            "max_score": max_score,
            "feedback": "\n".join(debug_lines),
        }

    solution_d_input = data["d_input"]
    if np.allclose(student_d_input, solution_d_input, rtol=1e-5, atol=1e-7):
        score += 5  # full points for backward pass.
        debug_lines.append("Backward pass gradients match expected gradients.")
    else:
        mean_diff, rel_err, max_diff = compute_error_metrics(
            student_d_input, solution_d_input
        )
        debug_lines.append(
            f"Backward pass gradients differ: Mean diff: {mean_diff:.3e}, relative error: {rel_err:.3e}, max diff: {max_diff:.3e}."
        )

    return {"score": score, "max_score": max_score, "feedback": "\n".join(debug_lines)}


def test_train():
    """
    Tests the end-to-end training function (train_end2end.py).
    Sets a fixed random seed, runs the training loop, captures printed epoch losses,
    and checks whether the loss improved over epochs.

    Scoring:
      - Full credit (10 points) if the final loss improves by at least 5%.
      - Partial credit (5 points) if there is some improvement.
      - No credit if the loss does not improve.
    Returns a dictionary with the score (max 10 points) and detailed feedback.
    """
    debug_lines = []
    np.random.seed(11785)

    try:
        train_module = importlib.import_module("train_end2end")
        if not hasattr(train_module, "train"):
            raise AttributeError(
                "Module 'train_end2end' does not have a 'train' function."
            )
        train_fn = getattr(train_module, "train")
        if not callable(train_fn):
            raise TypeError("'train' in module 'train_end2end' is not callable.")
        debug_lines.append("Successfully imported train_end2end.train.")
    except Exception as e:
        debug_lines.append(f"Error importing train_end2end: {e}")
        return {"score": 0, "max_score": 10, "feedback": "\n".join(debug_lines)}

    stdout_capture = io.StringIO()
    try:
        with redirect_stdout(stdout_capture):
            train_fn()
        debug_lines.append("Training function executed successfully.")
    except Exception as e:
        debug_lines.append(f"Error during training: {e}")
        return {"score": 0, "max_score": 10, "feedback": "\n".join(debug_lines)}

    output = stdout_capture.getvalue()

    # Parse printed lines for losses using a regex.
    loss_pattern = re.compile(r"Epoch\s+(\d+),\s+Loss:\s+([0-9\.eE+-]+)")
    losses = []
    for line in output.splitlines():
        match = loss_pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            losses.append((epoch, loss))

    if not losses:
        debug_lines.append("No loss values were printed during training.")
        return {"score": 0, "max_score": 10, "feedback": "\n".join(debug_lines)}

    losses.sort(key=lambda x: x[0])
    initial_loss = losses[0][1]
    final_loss = losses[-1][1]
    improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

    debug_lines.append("Parsed training losses:")
    for epoch, loss in losses:
        debug_lines.append(f"  Epoch {epoch}: {loss:.4f}")
    debug_lines.append(
        f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}"
    )
    debug_lines.append(f"Improvement: {improvement * 100:.1f}%")

    if improvement >= 0.05:
        score = 10
        debug_lines.append("Training loop improved loss sufficiently.")
    elif improvement > 0:
        score = 5
        debug_lines.append("Training loop improved loss slightly.")
    else:
        score = 0
        debug_lines.append("Training loop did not improve loss.")

    return {"score": score, "max_score": 10, "feedback": "\n".join(debug_lines)}


def run_all_tests():
    gru_result = test_gru()
    train_result = test_train()

    report_lines = []
    report_lines.append(
        "Problem: GRU (Score: {} / {})".format(
            gru_result["score"], gru_result["max_score"]
        )
    )
    if gru_result["feedback"]:
        report_lines.append("Feedback:")
        report_lines.append(gru_result["feedback"])
    report_lines.append("-" * 40)
    report_lines.append(
        "Problem: train_end2end (Score: {} / {})".format(
            train_result["score"], train_result["max_score"]
        )
    )
    if train_result["feedback"]:
        report_lines.append("Feedback:")
        report_lines.append(train_result["feedback"])
    report_lines.append("-" * 40)

    return {"GRU": gru_result, "train_end2end": train_result}, "\n".join(report_lines)


def main(args):
    results, report = run_all_tests()
    print("DETAILED REPORT:")
    print(report)
    print("FINAL SCORES:")
    print("GRU: {} / {}".format(results["GRU"]["score"], results["GRU"]["max_score"]))
    print(
        "train_end2end: {} / {}".format(
            results["train_end2end"]["score"], results["train_end2end"]["max_score"]
        )
    )


if __name__ == "__main__":
    main(sys.argv)
