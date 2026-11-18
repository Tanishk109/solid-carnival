# SpamScope

**Explainable English SMS spam classification for IS4158 NLP**

SpamScope is a focused, reproducible NLP investigation—not a collection of unrelated demos. It compares sparse, recurrent, and transformer-derived text representations on the official UCI SMS Spam Collection, explains linear predictions, calibrates probabilities, abstains on uncertain cases, and presents the evidence in a minimal Streamlit interface.

## Quick start

Python 3.9–3.12 is recommended. The trained artifacts are bundled, so the default hybrid TF-IDF model starts without retraining or downloading a model. If you select the optional MiniLM model, its public weights are downloaded and cached on first use.

### macOS / Linux

```bash
chmod +x run.sh
./run.sh
```

### Windows

Double-click `run.bat`, or run:

```bat
run.bat
```

Open `http://localhost:8765`. The first run creates `.venv` and installs the pinned dependencies.

## Verified results

The final test set contains 772 messages and is untouched until all model, calibration, decision-threshold, and review-threshold choices are frozen.

| Model | Macro-F1 | Spam recall | PR-AUC | Brier ↓ |
|---|---:|---:|---:|---:|
| Majority baseline | 0.468 | 0.000 | 0.122 | 0.107 |
| Word TF-IDF + LR | 0.956 | 0.894 | 0.968 | 0.015 |
| **Word/character TF-IDF + LR** | **0.967** | **0.957** | **0.975** | **0.012** |
| BiLSTM | 0.964 | 0.947 | 0.949 | 0.014 |
| MiniLM embeddings + LR | 0.953 | 0.872 | 0.972 | 0.014 |

This comparison is intentionally honest: frozen MiniLM embeddings did not win. Character patterns carry strong signal in short, noisy SMS. MiniLM is described accurately as a feature extractor plus logistic regression, not fine-tuning.

## What the interface demonstrates

- One-message analysis with four deployable models.
- Calibrated spam probability and model-specific decision threshold.
- A validation-selected manual-review state for low-confidence predictions.
- Exact TF-IDF × coefficient contributions for linear models.
- A concise test comparison with optional confusion matrices and ablations.
- Validation-only representation ablations.
- Thirty manually inspected errors with a failure taxonomy.
- Prominent scope and safety limitations.

The navigation is conditionally rendered. Unlike `st.tabs`, inactive pages do not execute expensive model or analysis code on every interaction.

## Reproduce and verify

From the activated environment:

```bash
python evaluate.py
python -m unittest discover -s tests -v
```

Expected outcome: the frozen evaluation audit passes and **32 tests pass**.

To retrain every model from the bundled data (MiniLM is downloaded on first use):

```bash
python train.py
```

The script reproduces the split, fits all models, calibrates on one half of validation, chooses decision/review thresholds on the other half, evaluates once on test, and overwrites the generated artifacts/results. After MiniLM has been cached locally, subsequent runs can work offline.

## Experimental protocol

1. Parse 5,574 labelled UCI messages.
2. Normalize URLs, email addresses, currency amounts, and phone numbers while preserving lexical and punctuation cues.
3. Remove 429 normalized duplicate rows **before** splitting.
4. Create a fixed-seed stratified 70/15/15 train/validation/test split.
5. Fit vocabularies, vectorizers, and classifiers using training only.
6. Split validation equally: sigmoid calibration on one half; decision and review thresholds on the other.
7. Evaluate frozen artifacts once on all 772 test examples.

The BiLSTM is trained under three predeclared seeds on the selection split to expose instability. Seed 4158 is the predeclared deployment artifact; the best run is not cherry-picked.

## Course mapping

| IS4158 concept | Where it appears |
|---|---|
| Cleaning and tokenization | `src/preprocess.py` |
| Bag-of-words, n-grams, TF-IDF | word and character pipelines |
| Supervised text classification | logistic regression and BiLSTM |
| Sequence modelling | compact bidirectional LSTM |
| Transformer representations | frozen `all-MiniLM-L6-v2` embeddings |
| Evaluation | precision, recall, F1, PR/ROC-AUC, calibration, confusion matrices, bootstrap CI |
| Applied NLP system | cached Streamlit inference UI using frozen artifacts |
| Critical model analysis | ablations, abstention, stability, and manual error taxonomy |

## Project structure

```text
spamscope_project/
├── app.py
├── train.py
├── evaluate.py
├── src/
│   ├── preprocess.py
│   ├── models.py
│   └── inference.py
├── data/
│   ├── SMSSpamCollection
│   ├── UCI_README.txt
│   ├── dataset_manifest.json
│   └── splits.csv
├── artifacts/
│   ├── models.joblib
│   └── bilstm_state.pt
├── models/                     # generated locally; ignored by Git
├── results/
│   ├── model_metrics.csv
│   ├── test_predictions.csv
│   ├── confusion_matrices.json
│   ├── ablation_results.csv
│   ├── bilstm_seed_results.csv
│   └── error_analysis.csv
├── tests/
├── MINI_PROJECT_REPORT.md
├── VIVA_CHEATSHEET.md
├── MODEL_CARD.md
├── requirements.txt
├── run.sh
└── run.bat
```

## Dataset and attribution

Almeida, T. A. and Hidalgo, J. M., *SMS Spam Collection*, UCI Machine Learning Repository, DOI `10.24432/C5CC84`. The repository lists the dataset under CC BY 4.0. Exact provenance, checksum, exclusions, counts, and the split policy are frozen in `data/dataset_manifest.json`.

MiniLM is downloaded from its upstream package and retains its Apache-2.0 license. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the dataset, derived-data,
and model notices. No license is granted for the original SpamScope source code
unless a top-level project license is added by the repository owner.

## Scope

SpamScope classifies historical English SMS as spam or ham. A ham prediction means only “not identified as spam.” It is not a phishing, malware, fraud, or safety detector, and it may fail under temporal, language, or adversarial shift.
