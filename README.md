# Semantic Contradiction Detector

## Overview
This project implements a **Semantic Contradiction Detector** as part of a technical assessment on *Deceptive Review Detection in the Wild*.  

The main goal is to detect **self-contradicting statements within a single product review**, which is a strong signal of low-quality or potentially deceptive reviews. Unlike traditional spam detection, this system focuses on **semantic meaning and logical consistency**, not just keywords.

---

## What This System Does
Given a single review, the system:
- Breaks the text into individual sentences
- Treats each sentence as a claim
- Compares claims to check if they contradict each other
- Returns:
  - Whether a contradiction exists
  - A confidence score (0–1)
  - The sentence pairs that contradict
  - A short explanation

---

## Approach Used
This implementation follows **Option A** from the assignment:

> **Sentence embeddings + semantic similarity with logical reasoning**

### How it works (in simple terms):
1. The review is split into sentences
2. Each sentence is converted into a semantic vector using **Sentence-BERT (SBERT)**
3. Sentence pairs are compared:
   - If they talk about the same topic  
   - But express opposite meanings (e.g., “fast” vs “slow”)
4. Such pairs are flagged as contradictions
5. A confidence score is calculated based on semantic similarity

This approach works well **without requiring any model training**, which makes it practical for real-world deployment with limited labeled data.

---

## Technologies Used
- Python 3.8+
- NumPy
- Scikit-learn
- Sentence-Transformers (SBERT)
- PyTorch (dependency of SBERT)
- Jupyter Notebook

---

## Installation (Jupyter Notebook)

Run the following command **inside a Jupyter cell**:

```python
%pip install numpy scikit-learn sentence-transformers torch
