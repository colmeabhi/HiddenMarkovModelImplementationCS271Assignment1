# CS-271-Assignment-1

## Professor - Mark Stamp

[Link to Assignment questions](https://www.cs.sjsu.edu/~stamp/CS271/syllabus/syllabusFall25.html#:~:text=Assignment%201%3A%20Due,AI%20generated%20music%3F)

Did:
    •	2.1: Computed P(O\mid\lambda) both by brute-force path Bruteforce solving and by the forward algorithm, then compared operation counts.
	•	2.3: Verified normalization by summing P(O) over all 3^4 sequences to get 1.0 and cross-checked forward DP against Bruteforce solving.
	•	2.10: Trained HMMs via Baum–Welch on cleaned English text for N=\{2,3,4,27\} and reported final (per-sequence) log-likelihoods.
	•	2.11: Built ciphertext from the corpus with a random shift, trained N=2 to spot vowel/consonant states, formed a bigram A (add-5), then ran N=M=26 with A frozen on 1000 chars to infer the substitution key and measured accuracy.

Code directories logic:
1. Main function file - hmmMain.py
2. Data handling - datasets.py
3. All the questions use hmmMain.py to import the functions.

