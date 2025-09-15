import os
import sys
import time
import numpy as np

from hmmMain import baumWelch
from datasets import loadCleanText, cleanLowerAlpha, bigramAFromCorpus, ALPHABET, IDX

def readCipherSymbols(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    tokens = [t for t in raw.split() if t.strip() != ""]
    if not tokens:
        raise ValueError("cipher token file is empty")
    uniq = []
    seen = set()
    for t in tokens:
        if t not in seen:
            uniq.append(t); seen.add(t)
    sym2id = {s: i for i, s in enumerate(uniq)}
    ids = np.array([sym2id[t] for t in tokens], dtype=int)
    return ids, sym2id, uniq

def readPlainLetters(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    text = cleanLowerAlpha(raw)
    if len(text) == 0:
        raise ValueError("plaintext file cleaned to empty (expected letters)")
    return text

def sampleFromText(text, L):
    # if shorter than L, just repeat, ok for bigrams
    if len(text) >= L:
        return text[:L]
    reps = L // max(1, len(text)) + 1
    return (text * reps)[:L]


# --- decoding / scoring ------------------------------------------------------

def mappingFromB(B):
    N, M = B.shape
    col_arg = np.argmax(B, axis=0)  # length M, entries in 0..N-1
    # build dict m -> letter
    out = {}
    for m in range(M):
        out[m] = ALPHABET[int(col_arg[m])]
    return out

def decodeCipher(O_sym_ids, map_m_to_char):
    return "".join(map_m_to_char[m] for m in O_sym_ids)

def accuracyPlain(decoded_letters, truth_letters):
    L = min(len(decoded_letters), len(truth_letters))
    if L == 0:
        return 0.0
    hits = 0
    for i in range(L):
        if decoded_letters[i] == truth_letters[i]:
            hits += 1
    return hits / float(L)

# --- main loop ---------------------------------------------------------------

def runZodiac408(
    cipher_path="zodiac408_cipher.txt",
    plain_path="zodiac408_plain.txt",
    corpus_path="BrownCorpus.txt",
    restarts=1000,
    iters=200,
    add_k=5,
    freeze_A=True,
    seed=7,
    bigram_chars=1_000_000,
):
    """
    Main loop: prepare A from bigrams, then do many BW restarts with random B, pi, Keep the best accuracy
    """
    t0 = time.time()

    # load cipher + gt plaintext
    O_sym, sym2id, id2sym = readCipherSymbols(cipher_path)
    plain = readPlainLetters(plain_path)
    plain = plain[:len(O_sym)] # trim to cipher length
    if len(plain) != len(O_sym):
        print("[warn] cipher/plain lengths differ after cleaning:", len(O_sym), len(plain))

    # build A from large English corpus + sample to length
    corpus = loadCleanText(corpus_path)
    corpus_big = sampleFromText(corpus, bigram_chars)
    A = bigramAFromCorpus(corpus_big, add_k=add_k)

    # model sizes
    N = 26                     # a to z
    M = len(id2sym)            # number of distinct cipher symbols
    T = len(O_sym)
    print(f"[info] N={N}, M={M}, T={T}, restarts={restarts}, iters={iters}, freeze_A={freeze_A}")

    # best across restarts
    best_acc = -1.0
    best_seed = None
    best_mapping = None

    # NOTE: pi/B are reinitialized each restart; A stays fixed if freeze_A=True
    rng = np.random.default_rng(seed)

    for r in range(restarts):
        # init random B, pi
        B0 = rng.random((N, M)); B0 = (B0.T / (B0.sum(axis=1) + 1e-12)).T
        pi0 = rng.random(N); pi0 = pi0 / (pi0.sum() + 1e-12)

        # train
        A_in = A if freeze_A else A.copy()
        A_learn, B_learn, pi_learn, ll = baumWelch(
            O_sym, A_in, B0, pi0,
            iters=iters, tol=1e-6, freeze_A=freeze_A
        )

        # decode + score
        m2c = mappingFromB(B_learn)
        decoded = decodeCipher(O_sym, m2c)
        acc = accuracyPlain(decoded, plain)

        if acc > best_acc:
            best_acc = acc
            best_seed = int(rng.bit_generator.state["state"]["state"])
            best_mapping = m2c

        # progress
        if (r + 1) % max(1, restarts // 10) == 0:
            print(f" : {r+1}/{restarts} restarts done (best={best_acc:.4f})")

    elapsed = time.time() - t0
    print(f"[done] best accuracy = {best_acc:.4f} over {restarts} restarts in toim {elapsed:.1f}s")
    return {
        "best_acc": best_acc,
        "best_seed_like": best_seed,
        "best_mapping": best_mapping,
        "A_used": A,
        "M": M,
        "T": T,
    }

if __name__ == "__main__":
    need = ["zodiac408_plain.txt", "zodiac408_cipher.txt", "BrownCorpus.txt"]
    for p in need:
        if not os.path.exists(p):
            print(f"[error] missing file: {p}")
            sys.exit(1)

    out = runZodiac408(
        cipher_path="zodiac408_cipher.txt",
        plain_path="zodiac408_plain.txt",
        corpus_path="BrownCorpus.txt",
        restarts=1000,        
        iters=200,
        freeze_A=True, # False for all other questions except 2.15 (a)
        bigram_chars=1000000,
    )
    print("best_acc (percent):", round(out["best_acc"] * 100, 2))