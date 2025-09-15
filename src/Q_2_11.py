import random
import numpy as np
from hmmMain import initRandom, baumWelch
from datasets import loadCleanText, encode, shiftText, bigramAFromCorpus, ALPHABET, IDX

def sampleFromText(text, L):
    if len(text) >= L:
        return text[:L]
    reps = L // max(1, len(text)) + 1
    return (text * reps)[:L]

def vowelStateIndex(B):
    vowels = "aeiou"
    best = -1.0; idx = 0
    for i in range(B.shape[0]):
        s = 0.0
        for v in vowels:
            s += float(B[i, IDX[v]])
        m = s / len(vowels)
        if m > best:
            best = m; idx = i
    return idx

def keyFromB(B):
    m = {}
    for p in range(26):
        row = B[p]
        j = int(np.argmax(row))
        m[ALPHABET[p]] = ALPHABET[j]
    return m

def fractionCorrect(mapping, shift_k):
    ok = 0
    for p in range(26):
        want = ALPHABET[(p + shift_k) % 26]
        if mapping[ALPHABET[p]] == want:
            ok += 1
    return ok / 26.0

if __name__ == "__main__":
    rng = random.Random(7)
    corpus = loadCleanText("BrownCorpus.txt")

    plaintext = sampleFromText(corpus, 50000)
    k = rng.randrange(26)
    ciphertext = shiftText(plaintext, k)
    O = np.array([IDX[ch] for ch in ciphertext], dtype=int)

    A2, B2, pi2 = initRandom(2, 26, seed=1)
    res1 = baumWelch(O, A2, B2, pi2, iters=200, tol=1e-6)
    A2 = res1[0]; B2 = res1[1]; pi2 = res1[2]
    v_state = vowelStateIndex(B2)
    print("[2.11b] vowel-like state index:", v_state)

    textForA = sampleFromText(corpus, 300000)
    A26 = bigramAFromCorpus(textForA, add_k=5)

    O1k = O[:1000]
    init = initRandom(26, 26, seed=3)
    B26 = init[1]; pi26 = init[2]
    A_fixed = A26.copy()

    res2 = baumWelch(O1k, A_fixed, B26, pi26, iters=200, tol=1e-6, freeze_A=True)
    Alearned = res2[0]; BLearned = res2[1]; pi_learned = res2[2]

    mapping = keyFromB(BLearned)
    acc = fractionCorrect(mapping, k)
    print("[2.11d] key accuracy:", round(acc, 4))