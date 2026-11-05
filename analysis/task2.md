# Task 2 Analysis: SelfCheckGPT Reproduction (WikiBio GPT-3)

## 1. Setup

- **Data.** 1,908 annotated sentences from 238 GPT-3 biographies, each with 20 supplied resamples (temperature 1.0). Minor and major inaccuracies are both positive (72.96% of sentences).
- **SelfCheck-BERTScore.** Each resample is split into sentences with spaCy. For a scored sentence, take the maximum BERTScore-F1 against the sentences of each resample, average over the 20 resamples, and return `1 - mean`. BERTScore uses `roberta-large` with the library defaults (layer 17, baseline rescaling).
- **SelfCheck-NLI.** The input is (sentence, whole resampled passage): the sentence is the hypothesis and the resample is the premise. The score is P(contradiction), averaged over the 20 resamples.
- **NLI checkpoint.** The assignment names `microsoft/deberta-v3-large-mnli`. I could not find a checkpoint with that ID, so `potsawee/deberta-v3-large-mnli` was used, which is the one in the SelfCheckGPT code. It has the neutral head removed, so P(contradiction) is already normalised over entail and contradict, as in the paper.
- **Metric.** AUC-PR (average precision), hallucinated = positive, on the full dataset. Higher scores mean more likely hallucinated for both variants.

## 2. Results

| Method | AUC-PR | ROC-AUC |
|---|---|---|
| Random baseline (positive rate) | 0.7296 | 0.5000 |
| SelfCheck-BERTScore | 0.8125 | 0.6484 |
| SelfCheck-NLI | 0.9250 | 0.8369 |

Score statistics:

| Label | BERTScore inconsistency (mean / median / std) | NLI contradiction (mean / median / std) |
|---|---|---|
| 0 (accurate) | 0.550 / 0.597 / 0.182 | 0.511 / 0.514 / 0.298 |
| 1 (hallucinated) | 0.637 / 0.657 / 0.135 | 0.850 / 0.927 / 0.198 |

Mean score by the original three-way annotation:

| Annotation | BERTScore inconsistency | NLI contradiction |
|---|---|---|
| accurate | 0.550 | 0.511 |
| minor inaccurate | 0.584 | 0.811 |
| major inaccurate | 0.681 | 0.883 |

## 3. Discussion

**Both variants beat the random baseline, and NLI does so by a wide margin.** BERTScore gains about 0.08 AUC-PR over the baseline but has a modest ROC-AUC of 0.648. NLI gains about 0.20 AUC-PR with a ROC-AUC of 0.837. On the PR curve, NLI holds precision around 0.95 or above up to roughly 40% recall, while BERTScore stays around 0.82 to 0.85.

**The results match the published ones.** The SelfCheckGPT repository reports about 81.96 (BERTScore) and 92.50 (NLI) AUC-PR for this setting. Our 0.8125 and 0.9250 are very close, which indicates a faithful reproduction: sentence splitting, the max-then-mean aggregation, the NLI input order and the checkpoint all behave as intended. Small gaps are normal from library versions and the spaCy model.

**NLI is stronger because it detects contradiction.** BERTScore measures only similarity of wording between the sentence and the resamples, so a hallucinated sentence about the same topic can still look similar. NLI asks whether a resample contradicts the claim, which is closer to what hallucination detection needs.

**The scores order the classes as the method predicts.** Both scores rise from accurate to minor to major inaccuracy (NLI: 0.511, 0.811, 0.883; BERTScore: 0.550, 0.584, 0.681), so major errors are the most inconsistent with the resamples.

**Comparison with Task 1.** Token probabilities (Task 1) were at or below the random baseline, while resampling consistency is strong. Probability measures fluency, which GPT-3's wrong sentences have. Agreement across independent generations is a much better proxy for factuality.

**Score distributions look as expected.**
- NLI is heavily skewed towards 1 (median 0.878), which is natural because most sentences in the dataset are hallucinated. This skew motivated the logit transform used in Task 3.
- BERTScore ranges from 0.0 to 1.03 because of baseline rescaling, so values slightly above 1 are normal.
- The thin vertical spike at recall 0 on the BERTScore PR curve comes from a few top-ranked sentences and carries no meaning.

## 4. Limitations

- **Resamples are fixed.** We use the 20 resamples shipped with the dataset and did not generate any ourselves, so results are tied to those samples.
- **BERTScore is similarity, not truth.** It cannot distinguish a faithful paraphrase from a plausible but wrong sentence, which limits its ceiling.
- **NLI model substitution.** The checkpoint differs from the ID named in the assignment (see Setup).
- **Binary label.** Minor and major inaccuracies are merged for AUC-PR. The three-way means above suggest major errors are easier to detect.
- **Cost.** Each sentence is compared with every resample (about 180 reference sentences for BERTScore, 20 whole passages for NLI), which is expensive compared with the probability baseline.

## 5. Conclusion

SelfCheckGPT is reproduced successfully on WikiBio: SelfCheck-NLI reaches 0.925 AUC-PR and SelfCheck-BERTScore 0.8125, against a random baseline of 0.7296, in line with the published results. NLI is the stronger and clearly useful detector. Both sets of scores are saved for Task 3, where they are combined with the Task 1 features.