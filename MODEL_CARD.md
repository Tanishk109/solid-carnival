# SpamScope Model Card

## Intended use

Educational demonstration of English SMS spam classification, model comparison, probability calibration, abstention, and explainability for an NLP course.

## Out-of-scope use

Do not use this project as a phishing, fraud, malware, moderation, or personal-safety system. Do not automatically punish senders or block essential messages using its output alone.

## Data

UCI SMS Spam Collection, DOI `10.24432/C5CC84`, CC BY 4.0. The project removes normalized duplicates and uses a fixed stratified train/validation/test split. Exact provenance and SHA-256 are recorded in `data/dataset_manifest.json`.

## Default model

Word and character TF-IDF feature union with class-weighted logistic regression, followed by held-out sigmoid calibration. Decision and review thresholds are selected on validation data.

## Performance

On the 772-example frozen test set: 0.986 accuracy, 0.967 macro-F1, 0.928 spam precision, 0.957 spam recall, 0.975 PR-AUC, 0.987 ROC-AUC, and 0.012 Brier score. Bootstrap 95% CI for macro-F1: [0.947, 0.985].

## Limitations

- Historical and primarily English data.
- Text-only: no sender reputation, redirect inspection, or campaign metadata.
- Possible corpus label ambiguity.
- Calibration may drift when class prevalence changes.
- Unknown performance on modern smishing, non-English text, images, and adversarial inputs.
- Explanations describe the linear model, not ground truth or human intent.

## Human oversight

Low-confidence cases are marked for manual review. A ham result is phrased as “not identified as spam” and is not a safety guarantee.
