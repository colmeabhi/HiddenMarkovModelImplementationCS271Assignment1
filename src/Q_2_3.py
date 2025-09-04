import itertools
import math
import numpy as np
from hmmMain import forwardScaled, bruteforceMethod

A = np.array([[0.7, 0.3],
              [0.4, 0.6]], dtype=float)
B = np.array([[0.1, 0.4, 0.5],
              [0.7, 0.2, 0.1]], dtype=float)
pi = np.array([0.0, 1.0], dtype=float)

def allSequences(M, T):
    for tup in itertools.product(range(M), repeat=T):
        yield np.array(tup, dtype=int)

if __name__ == "__main__":
    total_bf = 0.0
    total_fwd = 0.0

    for O in allSequences(3, 4):
        total_bf += bruteforceMethod(O, A, B, pi)
        res = forwardScaled(O, A, B, pi)
        total_fwd += math.exp(res[0])

    print("sum P(O) by Bruteforce solving:", round(total_bf, 12))
    print("sum P(O) by forward    :", round(total_fwd, 12))

    ok_bf = abs(total_bf - 1.0) < 1e-9
    ok_fwd = abs(total_fwd - 1.0) < 1e-9
    if ok_bf and ok_fwd:
        print("Checks passed.")
    else:
        print("Checks failed.")