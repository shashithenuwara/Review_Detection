# Semantic Contradiction Detector

## Project Overview
This project implements a **Semantic Contradiction Detector** as part of a technical assessment on *Deceptive Review Detection in the Wild*.  
The goal is to automatically identify **self-contradicting statements within a single product review**, which is a common signal of deceptive or low-trust reviews.

The system analyzes reviews at the sentence level using **semantic embeddings and logical heuristics** to determine whether contradictions exist.

---

## Key Features
- Sentence-level text preprocessing
- Claim extraction from reviews
- Semantic similarity using Sentence-BERT (SBERT)
- Detection of polarity and negation conflicts
- Confidence score generation (0–1)
- Identification of contradicting sentence pairs
- Evaluation using standard classification metrics

---

## Approach
This implementation follows **Option A** from the assignment:

> **Sentence embeddings + semantic similarity with logical reasoning**

### High-level steps:
1. Split the review into sentences
2. Encode each sentence using SBERT
3. Compare sentence pairs for:
   - High semantic similarity (same topic)
   - Opposing sentiment or negation
4. Flag contradictions and compute confidence
5. Evaluate performance on a synthetic dataset

The approach does **not require model training**, making it suitable for low-data and fast-deployment scenarios.

---

## Technologies Used
- Python 3.8+
- NumPy
- Scikit-learn
- Sentence-Transformers (SBERT)
- PyTorch (dependency of SBERT)
- Jupyter Notebook

---

## Installation (Jupyter Environment)

Run the following command **inside a Jupyter cell**:

```python
%pip install numpy scikit-learn sentence-transformers torch
