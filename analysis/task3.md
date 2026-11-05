# Task 3 Analysis: Stacked Ensemble Detector (WikiBio GPT-3)

## 1. Setup

- **Data.** 1,908 annotated sentences from 238 GPT-3 biographies. Minor and major inaccuracies are both the positive class, so 72.96% of all sentences are positive.
- **Split.** 80/20 by passage with a fixed seed of 42, giving 190 train and 48 test passages. No passage appears in both parts. The held-out part has a positive rate of 0.7624, which is the random-baseline AUC-PR there.
- **Features.**
  - Task 1: mean token log-probability and mean predictive entropy (Qwen3-8B, forced decoding, sentence scored in context).
  - Task 2: SelfCheck-BERTScore inconsistency and SelfCheck-NLI contradiction probability (20 supplied resamples).
- **Meta-classifier.** Standardised logistic regression fitted on the train part only. The regularisation strength `C` was chosen by 5-fold cross-validation grouped by passage, scored by average precision.
- **Improved variant.** The NLI probability was replaced by its logit (clipped at 1e-4) before fitting. This was added after seeing the first ensemble's results, so it is a post-hoc choice, supported by the cross-validation result below.
- **Metric.** AUC-PR (average precision), hallucinated = positive. All methods are evaluated on the same held-out 20%.

## 2. Results on the held-out 20%

| Method | AUC-PR | ROC-AUC |
|---|---|---|
| Random baseline (positive rate) | 0.7624 | 0.5000 |
| **Task 1** raw: -mean log-prob | 0.7101 | 0.4445 |
| **Task 1** raw: mean entropy | 0.7189 | 0.4438 |
| **Task 1** logistic regression (2 features) | 0.7754 | 0.5389 |
| **Task 2** raw: BERTScore | 0.8148 | 0.6260 |
| **Task 2** raw: NLI | 0.9306 | 0.8251 |
| **Task 2** logistic regression (2 features) | 0.8997 | 0.8070 |
| **Task 3** ensemble, all 4 features | 0.9073 | 0.7992 |
| **Task 3** ensemble, all 4 features with logit NLI | 0.9245 | 0.8150 |

Bootstrap over test passages (1,000 resamples, 95% intervals):

| Method | AUC-PR | 95% CI |
|---|---|---|
| Task 1 raw: mean entropy | 0.7189 | [0.6361, 0.8207] |
| Task 1 logistic regression | 0.7754 | [0.6980, 0.8582] |
| Task 2 raw: NLI | 0.9306 | [0.8897, 0.9648] |
| Task 2 logistic regression | 0.8997 | [0.8364, 0.9515] |
| Task 3 ensemble (all 4) | 0.9073 | [0.8531, 0.9522] |
| Task 3 ensemble (logit NLI) | 0.9245 | [0.8804, 0.9609] |

Paired bootstrap differences in AUC-PR:

| Comparison | Mean difference | 95% CI | Resamples favouring the first |
|---|---|---|---|
| Ensemble (all 4) minus Task 2 LR | +0.0066 | [-0.0097, +0.0246] | 78.7% |
| Logit ensemble minus ensemble (all 4) | +0.0165 | [+0.0052, +0.0305] | 100% |
| Logit ensemble minus raw NLI | -0.0061 | [-0.0127, -0.0007] | 0.9% |

Robustness check: grouped 5-fold cross-validation on all 238 passages (fixed `C = 1`, random baseline 0.7296).

| Method | AUC-PR |
|---|---|
| Task 1 logistic regression | 0.7489 |
| Task 2 logistic regression | 0.9000 |
| Task 3 ensemble (all 4) | 0.9059 |
| Task 3 ensemble (logit NLI) | 0.9232 |

## 3. Discussion

**Task 1 features are weak alone.** The raw probability scores are at or below the random baseline (ROC-AUC about 0.44). Hallucinated GPT-3 sentences are fluent and generic, so a proxy language model finds them as predictable as accurate ones, and probability tracks fluency, not truth. A logistic regression can still extract a little signal by learning the sign and weight of each feature (0.775 on the test split, 0.749 in cross-validation, both slightly above baseline), but the gain is small.

**Task 2 features carry most of the signal.** Resampling consistency does track factuality. NLI is far stronger than BERTScore, because it detects contradiction between a claim and the resamples, while BERTScore only measures similarity of wording.

**The plain ensemble gives a small, uncertain gain.** Adding the Task 1 features to the Task 2 features moves AUC-PR from 0.8997 to 0.9073 on the test split (0.9000 to 0.9059 in cross-validation). The bootstrap interval for the difference includes zero, so this gain is not reliable.

**Logit-transforming the NLI feature gives a clear improvement.** Raw NLI probabilities pile up near 0 and 1, which a linear model on the raw values handles poorly. The logit spreads out those saturated values. The improvement over the plain ensemble is +0.0165 on the test split with an interval that excludes zero, and +0.0173 in cross-validation (0.9232 against 0.9059).

**The ensemble does not beat zero-shot NLI.** Raw NLI scores 0.9306 against 0.9245 for the logit ensemble on the test split. The difference of -0.0061 has an interval that excludes zero, so raw NLI is slightly but consistently better there. On the full dataset, raw NLI (0.925, from Task 2) and the cross-validated logit ensemble (0.9232) are essentially tied. The probability features add no measurable gain beyond NLI in this setting.

## 4. Limitations

- **Small test set.** There are only 48 test passages (383 to 384 sentences), so differences in the third decimal place are noise. The cross-validation on all passages is a more stable estimate.
- **Bootstrap scope.** The bootstrap resamples test passages with the fitted models held fixed. It captures test-set noise, not variation from retraining.
- **Post-hoc variant.** The logit transform was chosen after seeing the first ensemble's results. Both ensembles are reported side by side for that reason.
- **Fixed `C` in cross-validation.** The cross-validation check uses `C = 1` for simplicity, whereas the test-split models tune `C`.
- **Proxy model.** Task 1 features come from Qwen3-8B, which did not write the text. Probabilities from the generating model might behave differently.

## 5. Conclusion

The Task 1 baseline is near-random, SelfCheckGPT's NLI variant is a strong zero-resource detector (AUC-PR about 0.93 against a 0.76 baseline on the test split), and a trained stack with a logit-transformed NLI feature reaches about 0.92. The stack clearly improves on the plain linear ensemble but matches, and does not exceed, raw NLI. In this setting, resampling consistency dominates and token-probability features add little.