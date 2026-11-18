# SpamScope Viva Cheatsheet

## 30-second opening

“SpamScope is an explainable English SMS spam classifier built on the official UCI corpus. My research question is whether classical sparse features, a BiLSTM, or frozen transformer embeddings work best for short noisy messages. I removed duplicates before a fixed 70/15/15 split, calibrated probabilities on half-validation, selected decision and abstention thresholds on the other half, and evaluated once on 772 test examples. A word-plus-character TF-IDF logistic model won with 0.967 macro-F1 and 0.957 spam recall. The result is useful because MiniLM was genuinely run but did not win; orthographic features matter in SMS.”

## Likely questions

### Why this project?

It is one focused classification problem with real labels, multiple syllabus-aligned representations, measurable hypotheses, and a deployable UI. Every visible feature is tied to evaluation.

### Why not use accuracy alone?

Only about 12.2% of deduplicated messages are spam. A majority model gets 87.8% accuracy while detecting zero spam. Macro-F1, spam recall, and PR-AUC reveal that failure.

### How did you prevent data leakage?

Normalized duplicates were removed before splitting. Vectorizers, vocabulary, and classifiers fit on training only. Calibration and threshold selection use separate halves of validation. Test is used only after choices are frozen.

### Why are there only 5,145 messages after loading 5,574?

There were 429 duplicate normalized texts. Keeping them could place the same campaign wording in train and test and inflate metrics.

### What does “hybrid” mean?

It is a feature union: word TF-IDF 1–2 grams plus character-within-word 3–5 grams, followed by one logistic-regression classifier. It is not an ensemble of probabilities.

### Why did character features help?

They represent spelling variation, short codes, punctuation, and word fragments that are common in noisy SMS and poorly captured by word tokens alone.

### Is MiniLM fine-tuned?

No. `all-MiniLM-L6-v2` is frozen and produces sentence embeddings. Logistic regression is trained on those vectors. The report calls this frozen transformer representation, not fine-tuning.

### Why did MiniLM not win?

Semantic embeddings can underweight orthographic cues such as premium numbers, abbreviated cost strings, URLs, and obfuscation. Its PR-AUC is high, but its selected-threshold spam recall is lower.

### How does the BiLSTM work?

It uses a vocabulary created from training only, 64-dimensional embeddings, a 64-unit bidirectional LSTM, dropout, a linear output, weighted binary cross-entropy, gradient clipping, and early stopping.

### Did you cherry-pick the best neural seed?

No. Seed 4158 was predeclared for deployment. Three seeds were run on the selection split; their macro-F1 mean is 0.941 with sample SD 0.017. The variation is reported.

### What is calibration?

Raw model scores are mapped to observed probabilities with sigmoid/Platt calibration on held-out validation data. Brier score and 10-bin ECE assess probability quality.

### Why are decision thresholds not always 0.5?

A calibrated probability threshold is selected on validation to maximize macro-F1. Class weighting and task trade-offs mean 0.5 is not automatically optimal.

### How is manual review selected?

For each model, validation searches for a minimum confidence that maximizes accepted-case accuracy while retaining at least 80% coverage. Test coverage and accepted accuracy are reported.

### Does abstention make errors disappear?

No. It separates uncertain cases for human review. For the hybrid model, coverage is 97.4% and accepted accuracy is 99.2%; the remaining 2.6% still requires a decision.

### What is the strongest result?

Hybrid word/character TF-IDF: 0.967 macro-F1, 0.957 spam recall, 0.975 PR-AUC, and 0.012 Brier score on 772 test examples.

### Is the difference statistically significant?

Bootstrap confidence intervals overlap, so the project does not claim definitive superiority from this sample alone. It reports the observed ranking and uncertainty.

### How are linear explanations calculated?

For active features, local contribution equals TF-IDF value × learned logistic coefficient. Positive values push toward spam; negative values push toward ham. Calibration changes the probability but not that linear ordering.

### Why not show token explanations for MiniLM or BiLSTM?

A coefficient explanation would be false for those architectures. The UI says that faithful token coefficients are unsupported instead of displaying a misleading explanation.

### What were the common errors?

Label ambiguity, conversational use of spam-like keywords, premium-rate spam disguised as personal chat, forwarded quotations that look bulk-sent, truncation, and multilingual/dialect noise.

### What is the biggest limitation?

Temporal and domain shift. This is a historical English SMS corpus. Good in-dataset performance does not prove performance on modern smishing or other languages.

### Is “ham” safe?

No. It only means the model did not identify the text as corpus-style spam. The UI deliberately says “not identified as spam,” never “safe.”

### How can the experiment be reproduced?

Run `python train.py`, then `python evaluate.py`, then `python -m unittest discover -s tests -v`. The dataset, split manifest, model weights, frozen encoder, results, and fixed seeds are bundled.

### What would you improve with more time?

Collect a current, multilingual, sender-aware dataset; annotate a truly independent modern smishing challenge set with multiple raters; tune cost-sensitive thresholds with stakeholders; and evaluate calibration under prevalence shift.

## Numbers to remember

- Source rows: **5,574**
- Unique rows: **5,145**
- Removed duplicates: **429**
- Train / validation / test: **3,601 / 772 / 772**
- Spam prevalence after deduplication: **12.2%**
- Hybrid macro-F1: **0.967**
- Hybrid spam recall: **0.957**
- Hybrid PR-AUC: **0.975**
- Hybrid coverage / accepted accuracy: **97.4% / 99.2%**
- Automated tests: **32**
- Manually categorized failures: **30**
