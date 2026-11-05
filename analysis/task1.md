# Task 1 Analysis: Probability-Based Hallucination Detection (WikiBio GPT-3)

## 1. Setup

- **Data.** 1,908 annotated sentences from 238 GPT-3 biographies (full dataset). Minor and major inaccuracies are both positive, so 72.96% of sentences are positive. This is the AUC-PR of a random scorer.
- **Model.** Qwen3-8B (bfloat16), used as a proxy: it did not write the text.
- **Scoring.** Forced decoding (teacher forcing). Each sentence is scored given a fixed prefix plus all earlier sentences of the same passage, and only the sentence's own tokens are scored. For each sentence we compute:
  - mean token log-probability,
  - mean predictive entropy (entropy of the full next-token distribution, averaged over the sentence's tokens).
- **Hallucination score.** The negative mean log-probability and the mean entropy (higher means more likely hallucinated). Nothing is trained.
- **Metric.** AUC-PR (average precision), hallucinated = positive, on the full dataset.

## 2. Results

| Score | AUC-PR | ROC-AUC |
|---|---|---|
| Random baseline (positive rate) | 0.7296 | 0.5000 |
| -mean log-probability | 0.6747 | 0.4332 |
| Mean entropy | 0.7194 | 0.4994 |

Score statistics by class (label 1 = hallucinated):

| Label | Mean log-prob (mean / median / std) | Mean entropy (mean / median / std) | Tokens per sentence |
|---|---|---|---|
| 0 (accurate) | -1.168 / -1.069 / 0.618 | 0.936 / 0.891 / 0.402 | 26.1 |
| 1 (hallucinated) | -1.011 / -0.944 / 0.472 | 0.919 / 0.880 / 0.349 | 26.0 |

## 3. Discussion

**Neither score beats the random baseline.** Mean entropy (0.7194) is essentially at chance, with ROC-AUC 0.4994. Mean log-probability (0.6747) is below the baseline, with ROC-AUC 0.4332.

**The log-probability signal points the wrong way.** Hallucinated sentences get a *higher* mean log-probability (-1.011) than accurate ones (-1.168), so Qwen finds the hallucinated sentences more predictable. The method's assumption (lower probability means more likely hallucinated) does not hold on this data. The score direction was kept as specified and not flipped to make the number look better.

**Likely explanation.** This is an interpretation, not something the experiment proves:
- GPT-3's wrong sentences are fluent and often generic ("He was a prolific author..."), which any language model predicts easily.
- Accurate sentences more often contain specific facts (names, dates, places) that a proxy model cannot predict, so they get lower probability.
- Probability and entropy therefore measure fluency and genericness, not factual correctness.

**Length is not the cause.** Both classes average about 26 tokens, so the effect is not a sentence-length artefact.

**Scoring in context did not rescue the baseline.** An earlier run that scored each sentence in isolation (no context) gave AUC-PR 0.6910 (log-probability) and 0.7155 (entropy), ROC-AUC 0.445 and 0.495. Adding context lowered the average log-probability by roughly half (from about -2.3 to about -1.1), as expected, but did not change the conclusion.

## 4. Limitations

- **Proxy model.** The probabilities come from Qwen3-8B, not from GPT-3, which generated the text. The original SelfCheckGPT work shows that probabilities from the generating model or a stronger proxy can behave differently, so this result is specific to this model and setup.
- **Context contains errors.** Earlier sentences in the context are GPT-3's own and may be wrong, which may affect later scores.
- **Binary label.** Minor and major inaccuracies are merged. Major errors may behave differently, but this was not analysed here.
- **Batching noise.** A batched versus one-at-a-time check showed small bfloat16 differences (below 0.01 in mean log-probability on the sentences checked), which does not affect the conclusions.

## 5. Conclusion

Token probability and entropy from a proxy LLM are not useful hallucination detectors on WikiBio: both are at or below the random baseline of 0.73 AUC-PR, and log-probability is inverted. This motivates the resampling-based method in Task 2. The features are still saved for Task 3, where a trained classifier can use them with learned signs and weights.