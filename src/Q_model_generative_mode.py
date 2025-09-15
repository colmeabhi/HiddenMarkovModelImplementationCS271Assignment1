import numpy as np
from hmmMain import initRandom, baumWelch
from datasets import loadCleanText, encode

def generateText(A, B, pi, length=100, seed=42):
    """
    Generate text using trained HMM parameters.
    """
    import numpy as np
    from datasets import ALPHABET
    
    rng = np.random.default_rng(seed)
    N, M = B.shape
    
    # Sample initial state
    state = rng.choice(N, p=pi)
    generated = []
    
    for t in range(length):
        # Emit character from current state
        char_idx = rng.choice(M, p=B[state, :])
        generated.append(ALPHABET[char_idx])
        
        # Transition to next state (except on last character)
        if t < length - 1:
            state = rng.choice(N, p=A[state, :])
    
    return ''.join(generated)

# Simple script to run part (d)
if __name__ == "__main__":
    import numpy as np
    from hmmMain import initRandom, baumWelch
    from datasets import loadCleanText, encode
    
    def sampleFromText(text, L):
        if len(text) >= L:
            return text[:L]
        reps = L // max(1, len(text)) + 1
        return (text * reps)[:L]
    
    # Load corpus
    corpus = loadCleanText("BrownCorpus.txt")
    training_text = sampleFromText(corpus, 100000)
    O = encode(training_text)
    
    print("Problem 2.15(d): HMM Text Generation")
    print("="*50)
    
    for N in [4, 8]:
        print(f"\nTraining HMM with N={N} states")
        
        # Initialize and train HMM
        A, B, pi = initRandom(N, 26, seed=7)
        A_trained, B_trained, pi_trained, logP = baumWelch(
            O, A, B, pi, iters=100, tol=1e-6, freeze_A=False
        )
        
        # Generate text
        generated = generateText(A_trained, B_trained, pi_trained, length=100, seed=42)
        
        print(f"Final log-likelihood: {logP:.2f}")
        print(f"Generated text (N={N}):")
        print(f"'{generated}'")
