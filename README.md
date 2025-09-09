# CS-271 Assignment 1

**Course:** CS 271 – Prof. Mark Stamp  
**Semester:** Fall 2025  

[Assignment Questions](https://www.cs.sjsu.edu/~stamp/CS271/syllabus/syllabusFall25.html#:~:text=Assignment%201%3A%20Due,AI%20generated%20music%3F)

---

## Completed
- **2.1**: Computed \(P(O|\lambda)\) via brute force and forward algorithm; compared operation counts.  
- **2.3**: Verified normalization by summing \(P(O)\) over all \(3^4\) sequences = 1.0; cross-checked with DP.  
- **2.10**: Trained HMMs with Baum–Welch on English text for \(N=\{2,3,4,27\}\); reported log-likelihoods.  
- **2.11**: Built ciphertext with random shift, trained \(N=2\) (vowel/consonant), formed bigram \(A\) (add-5), then ran \(N=M=26\) with \(A\) frozen to infer substitution key; measured accuracy.

---

## Structure
├── src/                  # Source files for each assignment question
│   ├── Q_2_1.py
│   ├── Q_2_3.py
│   ├── Q_2_10.py
│   └── Q_2_11.py
│
├── BrownCorpus.txt       # Text corpus used for HMM training
├── codeOutput.txt        # Saved sample output runs
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # License file
└── .gitignore            # Git ignore rules

---

## Usage
```bash
pip install -r requirements.txt

# Examples
python hmmMain.py --task forward_vs_bruteforce
python hmmMain.py --task normalization_check
python hmmMain.py --task train_text
python hmmMain.py --task cipher_decode

## Sample Output
P(O) by Bruteforce solving: 0.02488
P(O) by forward    : 0.02488
op counts, bf vs forward: 24 vs 18

sum P(O) by Bruteforce solving: 1.0
sum P(O) by forward    : 1.0
Checks passed.

N = 2 final logP = -285.99 chars = 109
N = 3 final logP = -280.58 chars = 109
N = 4 final logP = -265.48 chars = 109
N = 27 final logP = -142.0 chars = 109

[2.11b] vowel-like state index: 0
[2.11d] inferred key accuracy: 0.7692


