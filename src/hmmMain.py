import numpy as np

def normalizeRows(X):
    X = X.astype(float, copy=True)
    s = X.sum(axis=1, keepdims=True)
    s = s + (s == 0.0) * 1e-12
    return X / s

def initRandom(N, M, seed=0):
    rng = np.random.default_rng(seed)
    A = normalizeRows(rng.random((N, N)))
    B = normalizeRows(rng.random((N, M)))
    pi = rng.random(N)
    pi = pi / pi.sum()
    return A, B, pi

# Forward algorithm with scaling
def forwardScaled(O, A, B, pi):
    T = len(O); N = A.shape[0]
    alpha = np.zeros((T, N), dtype=float)
    c = np.zeros(T, dtype=float)

    obs0 = int(O[0])
    for i in range(N):
        alpha[0, i] = pi[i] * B[i, obs0]
    c[0] = alpha[0].sum() or 1.0
    alpha[0, :] = alpha[0, :] / c[0]

    for t in range(1, T):
        ot = int(O[t])
        for j in range(N):
            tot = 0.0
            for i in range(N):
                tot += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = tot * B[j, ot]
        c[t] = alpha[t].sum() or 1.0
        alpha[t, :] = alpha[t, :] / c[t]

    logP = np.log(c).sum()
    return logP, alpha, c

# Backward algorithm with scaling
def backwardScaled(O, A, B, c):
    T = len(O); N = A.shape[0]
    beta = np.zeros((T, N), dtype=float)
    beta[T - 1, :] = 1.0

    for t in range(T - 2, -1, -1):
        ot1 = int(O[t + 1])
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += A[i, j] * B[j, ot1] * beta[t + 1, j]
            beta[t, i] = s / c[t + 1]  # match forward scaling
    return beta

# Baum-Welch implementation
def baumWelch(O, A, B, pi, iters=50, tol=1e-6, freeze_A=False):
    N = A.shape[0]; M = B.shape[1]
    A = A.copy(); B = B.copy(); pi = pi.copy()
    last = None

    for _ in range(iters):
        fwd = forwardScaled(O, A, B, pi)
        logP, alpha, c = fwd[0], fwd[1], fwd[2]
        beta = backwardScaled(O, A, B, c)
        T = len(O)

        gamma = np.zeros((T, N), dtype=float)
        for t in range(T):
            row = 0.0
            for i in range(N):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                row += gamma[t, i]
            row = row or 1.0
            for i in range(N):
                gamma[t, i] = gamma[t, i] / row

        if not freeze_A:
            xi_sum = np.zeros((N, N), dtype=float)
            for t in range(T - 1):
                nxt = int(O[t + 1])
                denom = 0.0
                for i in range(N):
                    for j in range(N):
                        denom += alpha[t, i] * A[i, j] * B[j, nxt] * beta[t + 1, j]
                denom = denom or 1.0
                for i in range(N):
                    for j in range(N):
                        num = alpha[t, i] * A[i, j] * B[j, nxt] * beta[t + 1, j]
                        xi_sum[i, j] += num / denom
            A = normalizeRows(xi_sum)

        B_counts = np.zeros((N, M), dtype=float)
        for t in range(T):
            m = int(O[t])
            for j in range(N):
                B_counts[j, m] += gamma[t, j]
        B = normalizeRows(B_counts)

        for i in range(N):
            pi[i] = gamma[0, i]

        if last is not None and abs(logP - last) < tol:
            break
        last = logP

    return A, B, pi, last

# Brute-force method to calculate P(O|lambda)
def bruteforceMethod(O, A, B, pi):
    N = A.shape[0]; T = len(O)
    total = 0.0; states = list(range(N))

    def go(t, prob, prev):
        nonlocal total
        if t == 0:
            for s in states:
                total2 = prob * pi[s] * B[s, int(O[0])]
                go(1, total2, s)
        elif t < T:
            for s in states:
                total3 = prob * A[prev, s] * B[s, int(O[t])]
                go(t + 1, total3, s)
        else:
            total += prob

    go(0, 1.0, 0)
    return total

# Operation counts for bruteforce method
def opCountsBruteforce(N, T):
    return N * (N ** (T - 1)) * T

# Operation counts for forward algorithm
def op_counts_forward(N, T):
    return T * (N * N + N)