import numpy as np
from hmmMain import initRandom, baumWelch
from datasets import cleanLowerAlpha, encode

TEXT = """
I am abhishek and I am sleepy. I love machine learning and I love to solve problems. Sleeping is very necessary for good health.
"""

def run(N, chars=20000, iters=50, seed=7):
    text = cleanLowerAlpha(TEXT)[:chars]
    O = encode(text)
    A, B, pi = initRandom(N, 26, seed=seed)
    out = baumWelch(O, A, B, pi, iters=iters, tol=1e-6)
    A = out[0]; B = out[1]; pi = out[2]; logP = out[3]
    print("N =", N, "final logP =", round(logP, 2), "chars =", len(O))
    return A, B, pi, logP

if __name__ == "__main__":
    for N in [2, 3, 4, 27]:
        run(N, chars=20000, iters=50)