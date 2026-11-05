# Dataset 7.1 — WikiBio GPT-3 Hallucination Dataset
## Source
https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination (from the SelfCheckGPT paper, Manakul
et al., 2023).
## Format
Each example is one GPT-3-generated, Wikipedia-style biography passage, split into sentences, with a human accuracy
annotation per sentence, plus 20 independently resampled passages from the same prompt.
| Field               | Type           | Meaning                                                                           |
| ------------------- | -------------- | --------------------------------------------------------------------------------- |
| `gpt3_text`         | `string`       | The full generated biography passage.                                             |
| `gpt3_sentences`    | `list[string]` | The passage split into individual sentences.                                      |
| `annotation`        | `list[string]` | Per-sentence label: `accurate`, `minor_inaccurate`, or `major_inaccurate`.        |
| `gpt3_text_samples` | `list[string]` | 20 independently resampled full passages (temperature 1.0), used by SelfCheckGPT. |

## Splits
| Split | Size | Notes |
| ----- | ---- | ----- |
|full dataset|≈238 passages|No official split. If assigned this dataset, build your own fixed 80/20 train/test partition (stated seed) for Task 3’s meta-classifier; Tasks 1–2 use the full dataset directly (their scoring methods are formulas, not trained models).|
  
## Required Preprocessing
1. Binarize the 3-way annotation label to 2-way for AUC-PR: accurate = negative, both minor inaccurate/major inaccurate = positive.
2. Keep gpt3 sentences and annotation index-aligned.

## Task Usage
1. Task 1 use: If assigned this dataset: score every sentence with mean log-probability/entropy; report AUC-PR on
the full dataset.
2. Task 2 use: If assigned this dataset: score every sentence with SelfCheck-BERTScore/NLI against the 20
resamples; report AUC-PR on the full dataset.
3. Task 3 use: If assigned this dataset: build your 80/20 split; train the meta-classifier on the 80%; evaluate
AUC-PR on the held-out 20%.






# Dataset 7.2 — TruthfulQA

## Source

https://huggingface.co/datasets/truthful_qa, configuration generation. Load with `load_dataset("truthful_qa", "generation")`.

## Format

Each example is one question designed to elicit a common misconception, with acceptable correct answers and known-incorrect answers.

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question, often targeting a common misconception. |
| `correct_answers` | list[string] | Acceptable correct answers. |
| `incorrect_answers` | list[string] | Known-incorrect, plausible-sounding answers. |
| `category` | string | Topic category (38 categories total). |

## Splits

| Split | Size | Notes |
|---|---|---|
| validation | 817 questions | TruthfulQA’s entire usable dataset. If assigned this dataset, build your own fixed 80/20 train/test partition for Task 3, same convention as Dataset 7.1. |

## Required Preprocessing

1. Generate one free-text answer per question with your chosen open LLM, fixed decoding configuration.

2. Score the generated answer accurate if it matches/closely paraphrases any `correct_answers` entry, inaccurate otherwise. State your matching method (similarity threshold, or hand-verified sample) explicitly.

   - **Task 1 use:** If assigned this dataset: generate an answer per question, score with log-probability/entropy; report AUC-PR on the full dataset.

   - **Task 2 use:** If assigned this dataset: generate \(N = 20\) resamples per question; apply SelfCheck-BERTScore/NLI; report AUC-PR.

   - **Task 3 use:** If assigned this dataset: build your 80/20 split; train the meta-classifier on the 80%; evaluate AUC-PR on the held-out 20%.







# Dataset 7.3 — HaluEval (QA Subset)

## Source

https://github.com/RUCAIBox/HaluEval (Li et al., 2023); confirm the exact split name for the QA subset against whichever mirror you use.

## Format

Each example pairs one question with a correct answer and an LLM-generated plausible-but-wrong answer.

| Field | Type | Meaning |
|---|---|---|
| `question` | string | The question (originally drawn from an existing QA dataset). |
| `right_answer` | string | The correct answer. |
| `hallucinated_answer` | string | An LLM-generated plausible-sounding incorrect answer. |

## Splits

| Split | Size | Notes |
|---|---|---|
| QA subset | ≈10,000 pairs | No official split. If assigned this dataset, draw a fixed random sample for tractability if needed, and build your own 80/20 train/test partition for Task 3, same convention as Datasets 7.1–7.2. |

## Required Preprocessing

1. Score `right_answer` and `hallucinated_answer` directly with forced decoding (both are already given, unlike Datasets 7.1–7.2 where you generate an answer yourself).
2. Treat each pair as two separate scoring instances for AUC-PR purposes.

   - **Task 1 use:** If assigned this dataset: score both candidate answers per pair with log-probability/entropy; report AUC-PR.

   - **Task 2 use:** If assigned this dataset: generate resamples and apply SelfCheck-BERTScore/NLI (state your exact resampling protocol for this dataset); report AUC-PR.

   - **Task 3 use:** If assigned this dataset: build your 80/20 split; train the meta-classifier on the 80%; evaluate AUC-PR on the held-out 20%.