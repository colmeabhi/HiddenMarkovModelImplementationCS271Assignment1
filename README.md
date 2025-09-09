---

# CS 271: Hidden Markov Models

This repository contains the source code and materials for **Assignment 1** of CS 271, a course on cryptography and network security taught by **Prof. Mark Stamp** at San José State University during the Fall 2025 semester. The assignment focuses on implementing and applying **Hidden Markov Models (HMMs)** to solve various problems, including probability computation and a cryptographic task.

---

## Project Structure

The project is organized into a clear directory structure for easy navigation.

├── src/
│   ├── Q_2_1.py            # HMM probability computation (Brute Force vs. Forward Algorithm)
│   ├── Q_2_3.py            # Normalization check for the Forward Algorithm
│   ├── Q_2_10.py           # HMM training using the Baum-Welch algorithm on English text
│   └── Q_2_11.py           # Ciphertext decoding using an HMM
│
├── BrownCorpus.txt         # Text corpus used for HMM training in Q_2_10
├── codeOutput.txt          # Sample output from key program runs
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Files to ignore in Git


---

## Key Concepts and Implementation

This project explores several core concepts related to HMMs:

* **Problem 2.1**: We implemented and compared two methods for computing the probability of an observation sequence, $P(O|\lambda)$:
    * **Brute-Force**: This naive approach calculates all possible state paths, which is computationally expensive. We verified the high operation count.
    * **Forward Algorithm**: A dynamic programming approach that drastically reduces the number of operations, demonstrating its efficiency for HMM evaluation.
* **Problem 2.3**: We validated the **normalization property** of the Forward Algorithm by summing the probabilities of all possible observation sequences, confirming that the total probability equals 1.0. This ensures the model is well-formed.
* **Problem 2.10**: We used the **Baum-Welch algorithm** to train HMMs on the **Brown Corpus**, a large text dataset. We trained models with varying numbers of states ($N$)—2, 3, 4, and 27—and reported the final log-likelihoods. This demonstrates how HMMs can learn the statistical properties of language.
* **Problem 2.11**: We applied HMMs to a **cryptanalysis** problem. The task involved decoding a ciphertext created with a substitution cipher. We used a two-step approach:
    1.  Trained an $N=2$ HMM to distinguish vowels from consonants.
    2.  Used this knowledge to initialize an $N=M=26$ HMM with a frozen transition matrix to infer the substitution key.

---

## Getting Started

### Prerequisites

You need **Python 3.x** installed to run this project.

### Installation

First, clone the repository to your local machine:

```bash
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo
Next, install the required Python libraries using the requirements.txt file:

Bash
pip install -r requirements.txt
Usage

The main entry point for all programs is hmmMain.py. You can run a specific task using the --task flag.

Bash
# Calculate P(O|λ) using both methods and compare operation counts
python src/hmmMain.py --task forward_vs_bruteforce

# Verify the normalization of the Forward Algorithm
python src/hmmMain.py --task normalization_check

# Train HMMs on English text with different state counts
python src/hmmMain.py --task train_text

# Decode a ciphertext using the HMM approach
python src/hmmMain.py --task cipher_decode
Sample Output
A sample of the program's output has been saved to codeOutput.txt for your convenience. This file demonstrates the expected results for each task.

Forward vs. Brute-Force:

P(O) by Bruteforce solving: 0.02488
P(O) by forward    : 0.02488
op counts, bf vs forward: 24 vs 18
Normalization Check:

sum P(O) by Bruteforce solving: 1.0
sum P(O) by forward    : 1.0
Checks passed.
Text Training:

N = 2 final logP = -285.99 chars = 109
N = 3 final logP = -280.58 chars = 109
N = 4 final logP = -265.48 chars = 109
N = 27 final logP = -142.0 chars = 109
Cipher Decoding:

[2.11b] vowel-like state index: 0
[2.11d] inferred key accuracy: 0.7692
