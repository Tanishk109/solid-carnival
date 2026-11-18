# SpamScope: Explainable English SMS Spam Classification

## Abstract

SpamScope investigates a narrow question: **how do sparse lexical features, a recurrent neural network, and frozen transformer embeddings compare on short, noisy SMS spam classification?** The project uses the official UCI SMS Spam Collection, removes duplicate messages before splitting, and evaluates five models on the same untouched test set. It adds sigmoid probability calibration, validation-selected abstention, interpretable token contributions, an ablation study, three-seed BiLSTM stability analysis, and a manual taxonomy of 30 failures. The strongest system is a word/character TF-IDF logistic-regression model with 0.967 macro-F1, 0.957 spam recall, and 0.975 PR-AUC. This result also demonstrates that a transformer representation is not automatically superior when orthographic patterns are central to the task.

## 1. Problem and research questions

SMS spam is imbalanced and linguistically noisy. Messages use shortened words, unusual punctuation, premium-rate numbers, URLs, currency amounts, and deliberate misspellings. Accuracy alone can conceal a classifier that predicts the majority ham class.

The investigation asks:

1. Do word bigrams improve over unigram TF-IDF?
2. Do character n-grams improve ranking and final test performance on noisy messages?
3. How does a compact BiLSTM compare with linear sparse models?
4. Do frozen MiniLM sentence embeddings outperform task-specific sparse features?
5. Can calibrated abstention improve reliability on automatically accepted predictions?

The application is deliberately limited to this task. It does not present generic sentiment, regex entity extraction, summarization, or retrieval as if those features answered the same research question.

## 2. Dataset and provenance

The SMS Spam Collection contains 5,574 English SMS messages labelled `ham` or `spam`. It is distributed by the UCI Machine Learning Repository under CC BY 4.0 (DOI `10.24432/C5CC84`). The exact downloaded file has SHA-256:

`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`

After Unicode-aware normalization, 429 duplicate rows were removed. No conflicting duplicate labels were found, leaving 5,145 unique messages: 4,517 ham and 628 spam. Deduplication occurs before splitting so repeated campaign messages cannot inflate performance by appearing in both train and test.

Using seed 4158, the remaining messages are stratified into:

- Train: 3,601 (70%)
- Validation: 772 (15%)
- Test: 772 (15%)

The validation set is divided in half. One half fits a sigmoid/Platt calibrator. The other selects the decision threshold and manual-review confidence threshold. The test labels are consulted only after those choices are frozen.

## 3. Preprocessing and representations

Normalization lowercases Unicode-normalized text, collapses whitespace, and replaces URLs, email addresses, currency amounts, and long phone patterns with typed placeholders. It intentionally retains wording and punctuation because they may carry class signal.

Five systems are evaluated:

1. **Majority-class baseline.** Always predicts the training majority class; its constant score equals training spam prevalence.
2. **Word TF-IDF + logistic regression.** Unigrams and bigrams, sublinear term frequency, class weighting, and a linear classifier.
3. **Hybrid word/character TF-IDF + logistic regression.** A feature union of word 1–2 grams and character-within-word 3–5 grams. “Hybrid” means feature union, not probability ensembling.
4. **BiLSTM.** A train-only vocabulary, 64-dimensional learned embeddings, a 64-unit bidirectional LSTM, dropout, gradient clipping, class-weighted binary loss, and early stopping.
5. **MiniLM embeddings + logistic regression.** Frozen `sentence-transformers/all-MiniLM-L6-v2` sentence vectors with a trained linear classifier. This is not transformer fine-tuning.

Every deployable model is actually executed. The Streamlit UI loads the same serialized artifacts that produced the reported metrics.

## 4. Evaluation design

The primary metric is macro-F1 because it gives equal importance to ham and spam performance. Supporting metrics are accuracy, spam precision/recall/F1, PR-AUC, ROC-AUC, Brier score, 10-bin expected calibration error, median single-message latency, confusion matrix, and artifact size. Macro-F1 includes a 500-resample bootstrap 95% confidence interval.

For selective prediction, each model sends cases below a validation-selected confidence threshold to manual review. Coverage is the fraction automatically accepted; accepted accuracy is measured only on those retained cases. Thresholds are model-specific.

## 5. Final test results

| Model | Accuracy | Macro-F1 | 95% CI | Spam P | Spam R | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority | 0.878 | 0.468 | [0.461, 0.474] | 0.000 | 0.000 | 0.122 | 0.107 |
| Word TF-IDF + LR | 0.982 | 0.956 | [0.932, 0.977] | 0.955 | 0.894 | 0.968 | 0.015 |
| **Word/character TF-IDF + LR** | **0.986** | **0.967** | **[0.947, 0.985]** | 0.928 | **0.957** | **0.975** | **0.012** |
| BiLSTM | 0.984 | 0.964 | [0.943, 0.984] | 0.927 | 0.947 | 0.949 | 0.014 |
| MiniLM + LR | 0.981 | 0.953 | [0.927, 0.976] | **0.965** | 0.872 | 0.972 | 0.014 |

The hybrid sparse model is selected as the default because it has the best macro-F1, spam recall, PR-AUC, and Brier score. Its automatic coverage is 0.974, and accuracy among accepted predictions is 0.992 versus 0.986 without abstention. This is a selective-prediction result, not a claim that abstained messages disappear.

The BiLSTM is close to the hybrid model, but its selection macro-F1 across seeds 4158, 4159, and 4160 is 0.952, 0.950, and 0.921 (mean 0.941, sample standard deviation 0.017). This exposes instability that a single best run would hide.

MiniLM has high ranking quality but lower recall at its selected threshold. Manual errors show why: sentence semantics can underweight premium-rate numbers, unusual spelling, stop-text instructions, and other orthographic cues. The result is not anti-transformer; it is evidence that representation choice should match the task.

## 6. Ablation study

Ablations are evaluated on the selection half of validation, not test.

| Variant | Macro-F1 | Spam F1 | PR-AUC |
|---|---:|---:|---:|
| Word unigrams, normalized | 0.941 | 0.896 | 0.951 |
| Word 1–2 grams, normalized | **0.952** | **0.915** | 0.964 |
| Word 1–2 grams, raw surface | 0.944 | 0.901 | 0.958 |
| Word + character, normalized | 0.945 | 0.903 | **0.975** |

Bigrams improve the fixed-0.5 classification score. Typed normalization improves over raw surface text. Character features produce the best PR-AUC, meaning the ranking improves even though a naive 0.5 threshold is not optimal; the deployed hybrid threshold is therefore selected on validation rather than assumed.

## 7. Error analysis

Thirty misclassifications were manually inspected across the four deployable models. The largest categories were label ambiguity/noise, conversational keyword ambiguity, obfuscated premium-rate spam, and legitimate messages with a bulk-message style (four each). Other failures include truncated context, short messages, URL-led promotions, service notifications, Romanized multilingual text, and chain warnings.

Important patterns are:

- Some corpus spam labels look like jokes or personal conversation; text alone cannot recover campaign or sender metadata.
- Linear models can overreact to `call`, `phone`, `win`, or brand names in legitimate messages.
- MiniLM can understand the conversational frame but miss short-code, cost, and obfuscation signals.
- Legitimate gateway footers and forwarded quotations resemble bulk spam.
- False negatives remain costly even when aggregate metrics are high.

The full message-level audit is in `results/error_analysis.csv` and is filterable in the app.

## 8. Application engineering

The Streamlit interface uses a conditional segmented-page router rather than tabs, so inactive dashboards and error tables do not execute during live inference. Model and tabular artifacts are cached. MiniLM is loaded only when selected. A form prevents rerun ambiguity, empty text is rejected, input length is bounded, and submitted messages are not written to disk.

Linear feature contributions are computed as TF-IDF value multiplied by logistic-regression coefficient. For BiLSTM and MiniLM, the application explicitly refuses to claim faithful token coefficients instead of displaying a misleading explanation.

## 9. Limitations and ethics

The corpus is historical, English-heavy, and much smaller than a production dataset. Modern smishing, multilingual messages, images, sender reputation, links, and adversarial spellings can cause distribution shift. Class labels do not establish malicious intent. Calibration can drift when real spam prevalence changes. Manual review transfers work rather than eliminating error.

The interface therefore uses “not identified as spam,” never “safe.” It is an educational classifier, not a security product, and must not be used as the sole basis for blocking communication or taking action against a sender.

## 10. Conclusion

SpamScope demonstrates that careful experimental control can be more impressive than feature count. It integrates preprocessing, n-grams, TF-IDF, supervised classification, recurrent networks, transformer-derived embeddings, calibration, explainability, evaluation, error analysis, and application development around one defensible question. On this dataset, character-aware sparse features provide the best balance of spam recall, calibration, speed, and interpretability.
