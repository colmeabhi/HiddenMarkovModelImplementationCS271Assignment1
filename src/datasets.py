import numpy as np
import re
from hmmMain import normalizeRows

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
IDX = {ch: i for i, ch in enumerate(ALPHABET)}

def cleanLowerAlpha(text):
    return re.sub(r'[^a-z]', '', text.lower())

def loadCleanText(path):
    f = open(path, "r", encoding="utf-8", errors="ignore")
    raw = f.read()
    f.close()
    return cleanLowerAlpha(raw)

def encode(text):
    out = []
    for ch in text:
        if ch in IDX:
            out.append(IDX[ch])
    return np.array(out, dtype=int)

def shiftText(text, k):
    out = []
    for ch in text:
        if ch in IDX:
            j = (IDX[ch] + k) % 26
            out.append(ALPHABET[j])
    return "".join(out)

def bigramAFromCorpus(text, add_k=5):
    txt = cleanLowerAlpha(text)
    counts = np.full((26, 26), float(add_k))
    for a, b in zip(txt, txt[1:]):
        i = IDX[a]; j = IDX[b]
        counts[i, j] += 1.0
    return normalizeRows(counts)