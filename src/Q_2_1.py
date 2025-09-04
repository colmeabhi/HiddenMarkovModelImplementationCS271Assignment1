import math
import numpy as np
from hmmMain import forwardScaled, bruteforceMethod, opCountsBruteforce, op_counts_forward

# Initialization from the book, A, B, pi, O are standard notations

A = np.array([[0.7, 0.3],
              [0.4, 0.6]], dtype=float)
B = np.array([[0.1, 0.4, 0.5],
              [0.7, 0.2, 0.1]], dtype=float)
pi = np.array([0.0, 1.0], dtype=float)
O = np.array([1, 0, 2], dtype=int)

if __name__ == "__main__":
    bf = bruteforceMethod(O, A, B, pi)
    res = forwardScaled(O, A, B, pi)
    logP = res[0]
    p_fwd = math.exp(logP)

    print("P(O) by Bruteforce solving:", round(bf, 9))
    print("P(O) by forward    :", round(p_fwd, 9))
    print("op counts, bf vs forward:", opCountsBruteforce(2, 3), "vs", op_counts_forward(2, 3))