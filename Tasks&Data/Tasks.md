# Project 7: Do LLMs Know When They’re Making Things Up?

## Problem Statement

A language model hallucinates when it generates fluent, confident text that is factually wrong. Because the generated text gives no obvious surface signal that anything is wrong, detecting hallucination requires looking beyond fluency.

The simplest signal is the model’s own token probabilities: lower probability or higher predictive entropy when generating something it is less sure of. Task 1 builds this baseline directly, on your assigned dataset.

Task 2 reproduces a more expensive but zero-resource idea: generate several independent resamples of the same passage, and check whether the sentence being scored is consistent with those resamples, measured via BERTScore and an NLI contradiction probability.

Task 3 combines both signals into one trained detector: split your assigned dataset’s items into an 80/20 train/test partition (fixed seed), fit a logistic-regression meta-classifier on the 80% using Task 1 + Task 2’s features, and evaluate on the held-out 20%.

Results are reported as AUC-PR throughout (the standard metric when the interesting class – hallucinations – is a minority of examples).

## Datasets Used

(Full dataset details are in the Datasets document, Section 7.)

Your group will be assigned exactly one of the three datasets below, and will complete all three tasks (baseline, paper reproduction, combine & improve) using only that dataset – you will not use the other two.

1. **WikiBio GPT-3 Hallucination Dataset**: GPT-3-generated biography passages with per-sentence accuracy annotations and pre-generated resamples. Sufficient on its own for all three tasks.

2. **TruthfulQA**: Adversarially-constructed factual questions with correct/incorrect answer lists. Sufficient on its own for all three tasks.

3. **HaluEval (QA subset)**: Question-answer pairs with a right answer and a hallucinated answer. Sufficient on its own for all three tasks.

# Task 1 — Baseline: Probability-Based Hallucination Detection (30 marks)

## Objective

Score every item in your assigned dataset for likely hallucination using only the model’s own token probabilities.

## Step-by-step Instructions

1. Using an open LLM, run a forced/greedy decoding pass over each item in your assigned dataset and compute mean token log-probability and mean predictive entropy.
2. Use these as the hallucination score (lower log-probability / higher entropy ⇒ more likely hallucinated).
3. Report sentence/answer-level AUC-PR against your assigned dataset’s own accurate/inaccurate ground truth (see companion Datasets document for how each dataset defines this label).

## Evaluation Metrics to Report

- AUC-PR on your assigned dataset.

---

# Task 2 — Paper Reproduction: SelfCheckGPT (30 marks)

## Objective

Reproduce a zero-resource, resampling-based hallucination detector on your assigned dataset.

**Paper:** Manakul, Liusie, Gales, *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*, arXiv:2303.08896.

**Code:** https://github.com/potsawee/selfcheckgpt

## Step-by-step Instructions

1. Implement SelfCheck-BERTScore: max BERTScore-F1 (backbone roberta-large) between each scored item and N = 20 resampled passages (use the dataset’s own resamples if provided, as in WikiBio) or generate N = 20 yourself at temperature 1.0, as for TruthfulQA/HaluEval.
2. Implement SelfCheck-NLI: P(contradiction) from microsoft/deberta-v3-large-mnli between each scored item and the same resamples.
3. Report AUC-PR (both variants) against your assigned dataset’s ground truth.

## Evaluation Metrics to Report

- AUC-PR (both SelfCheck variants) on your assigned dataset.

---

# Task 3 — Combine & Improve: Stacked Ensemble Detector (40 marks)

## Objective

Combine Task 1 and Task 2’s features into a single trained detector, on your assigned dataset.

## Step-by-step Instructions

1. Split your assigned dataset into an 80/20 train/test partition with a fixed, stated random seed.
2. Train a logistic-regression meta-classifier on the 80% train portion, using both Task 1’s features (log-probability, entropy) and Task 2’s features (BERTScore-inconsistency, NLI-contradiction) as inputs.
3. Evaluate AUC-PR on the held-out 20%, comparing Task 1 alone, Task 2 alone, and this Task 3 ensemble (recomputed on the same 20% for a fair comparison).

## Evaluation Metrics to Report

- AUC-PR on your assigned dataset’s held-out 20%, comparing Task 1, Task 2, and Task 3.

# References
Manakul, Liusie, Gales, arXiv:2303.08896. Lin, Hilton, Evans, TruthfulQA: Measuring How Models Mimic Human
Falsehoods, arXiv:2109.07958. Li et al., HaluEval: A Large-Scale Hallucination Evaluation Benchmark,
arXiv:2305.11747.