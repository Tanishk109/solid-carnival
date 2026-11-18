#!/usr/bin/env python3
"""Train every reported SpamScope model and freeze reproducible artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models import (
    BiLSTMConfig,
    build_hybrid_model,
    build_vocabulary,
    build_word_model,
    calibrate,
    choose_review_confidence,
    choose_threshold,
    encode_text,
    expected_calibration_error,
    fit_platt_calibrator,
    make_bilstm,
    seed_everything,
)
from src.preprocess import deduplicate, load_uci_sms, stratified_split


ROOT = Path(__file__).resolve().parent
SEED = 4158


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_macro_f1(labels: np.ndarray, predictions: np.ndarray, repeats: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    scores: list[float] = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        scores.append(float(f1_score(labels[indices], predictions[indices], average="macro")))
    return tuple(float(value) for value in np.quantile(scores, [0.025, 0.975]))


def metric_row(
    model_key: str,
    model_name: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    review_confidence: float,
    latency_ms: float,
    artifact_size_mb: float,
) -> tuple[dict[str, float | str], list[list[int]]]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, spam_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    macro_f1 = float(f1_score(labels, predictions, average="macro"))
    low, high = bootstrap_macro_f1(labels, predictions)
    confidence = np.maximum(probabilities, 1 - probabilities)
    retained = confidence >= review_confidence
    row: dict[str, float | str] = {
        "model_key": model_key,
        "model": model_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": macro_f1,
        "macro_f1_ci_low": low,
        "macro_f1_ci_high": high,
        "spam_precision": float(precision),
        "spam_recall": float(recall),
        "spam_f1": float(spam_f1),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece_10_bin": expected_calibration_error(labels, probabilities),
        "decision_threshold": threshold,
        "review_confidence": review_confidence,
        "coverage": float(retained.mean()),
        "accepted_accuracy": float(accuracy_score(labels[retained], predictions[retained])) if retained.any() else float("nan"),
        "median_latency_ms": latency_ms,
        "artifact_size_mb": artifact_size_mb,
        "test_examples": int(len(labels)),
    }
    return row, confusion_matrix(labels, predictions, labels=[0, 1]).tolist()


def median_latency(predict_one: Callable[[str], float], texts: list[str], runs: int = 25) -> float:
    samples = texts[: min(runs, len(texts))]
    timings: list[float] = []
    for text in samples:
        started = time.perf_counter()
        predict_one(text)
        timings.append((time.perf_counter() - started) * 1_000)
    return float(np.median(timings))


def fit_deployable_sklearn(model, train_x, train_y, calibration_x, calibration_y, selection_x, selection_y):
    model.fit(train_x, train_y)
    calibration_raw = model.predict_proba(calibration_x)[:, 1]
    calibrator = fit_platt_calibrator(calibration_raw, calibration_y)
    selection_probability = calibrate(model.predict_proba(selection_x)[:, 1], calibrator)
    threshold = choose_threshold(selection_y, selection_probability)
    review = choose_review_confidence(selection_y, selection_probability, threshold)
    return {
        "estimator": model,
        "calibrator": calibrator,
        "decision_threshold": threshold,
        "review_confidence": review,
    }


def _collate(sequences: list[list[int]], labels: list[int] | None = None):
    import torch

    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    width = int(lengths.max())
    tokens = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    if labels is None:
        return tokens, lengths
    return tokens, lengths, torch.tensor(labels, dtype=torch.float32)


def bilstm_probabilities(network, texts: list[str], vocab: dict[str, int], max_length: int, batch_size: int = 128) -> np.ndarray:
    import torch

    network.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            sequences = [encode_text(text, vocab, max_length=max_length) for text in texts[start : start + batch_size]]
            tokens, lengths = _collate(sequences)
            output.append(torch.sigmoid(network(tokens, lengths)).cpu().numpy())
    return np.concatenate(output)


def train_bilstm(
    train_x: list[str],
    train_y: np.ndarray,
    selection_x: list[str],
    selection_y: np.ndarray,
    *,
    seed: int = SEED,
    epochs: int = 8,
    max_length: int = 80,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class SmsDataset(Dataset):
        def __init__(self, texts, labels, vocab):
            self.sequences = [encode_text(text, vocab, max_length=max_length) for text in texts]
            self.labels = [int(label) for label in labels]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.sequences[index], self.labels[index]

    seed_everything(seed)
    vocab = build_vocabulary(train_x)
    config = BiLSTMConfig(vocab_size=len(vocab))
    network = make_bilstm(config)
    negative, positive = np.bincount(train_y, minlength=2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32))
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=1e-4)
    dataset = SmsDataset(train_x, train_y, vocab)
    generator = torch.Generator().manual_seed(seed)

    def collate_batch(batch):
        sequences, labels = zip(*batch)
        return _collate(list(sequences), list(labels))

    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_batch, generator=generator)
    best_score = -1.0
    best_state = None
    patience = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        network.train()
        losses: list[float] = []
        for tokens, lengths, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = network(tokens, lengths)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        selection_raw = bilstm_probabilities(network, selection_x, vocab, max_length)
        selection_score = float(f1_score(selection_y, selection_raw >= 0.5, average="macro"))
        history.append({"seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)), "selection_macro_f1": selection_score})
        if selection_score > best_score + 1e-4:
            best_score = selection_score
            best_state = copy.deepcopy(network.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= 2:
                break
    if best_state is None:
        raise RuntimeError("BiLSTM training did not produce a checkpoint")
    network.load_state_dict(best_state)
    network.eval()
    return network, vocab, config, history


def evaluate_ablation(name: str, estimator, train_x, train_y, selection_x, selection_y) -> dict[str, float | str]:
    estimator.fit(train_x, train_y)
    probability = estimator.predict_proba(selection_x)[:, 1]
    prediction = probability >= 0.5
    return {
        "variant": name,
        "evaluation_split": "selection",
        "macro_f1": float(f1_score(selection_y, prediction, average="macro")),
        "spam_f1": float(f1_score(selection_y, prediction, average="binary")),
        "pr_auc": float(average_precision_score(selection_y, probability)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "SMSSpamCollection")
    parser.add_argument("--transformer", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--skip-transformer", action="store_true", help="Development only; final submission includes it")
    parser.add_argument("--skip-bilstm", action="store_true", help="Development only; final submission includes it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(SEED)
    artifacts_dir = ROOT / "artifacts"
    results_dir = ROOT / "results"
    models_dir = ROOT / "models"
    for directory in (artifacts_dir, results_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw = load_uci_sms(args.data)
    clean, dedupe_audit = deduplicate(raw)
    frame = stratified_split(clean, seed=SEED)
    frame.to_csv(ROOT / "data" / "splits.csv", index=False)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    validation = frame[frame["split"] == "validation"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    calibration_indices, selection_indices = train_test_split(
        validation.index,
        test_size=0.5,
        random_state=SEED,
        stratify=validation["label"],
    )
    calibration_frame = validation.loc[calibration_indices].reset_index(drop=True)
    selection_frame = validation.loc[selection_indices].reset_index(drop=True)

    train_x = train["normalized_text"].tolist()
    train_y = train["label"].to_numpy()
    calibration_x = calibration_frame["normalized_text"].tolist()
    calibration_y = calibration_frame["label"].to_numpy()
    selection_x = selection_frame["normalized_text"].tolist()
    selection_y = selection_frame["label"].to_numpy()
    test_x = test["normalized_text"].tolist()
    test_y = test["label"].to_numpy()

    model_bundle: dict[str, dict] = {}
    raw_test_probabilities: dict[str, np.ndarray] = {}
    latencies: dict[str, float] = {}
    sizes: dict[str, float] = {}

    print("[1/4] Training word TF-IDF model", flush=True)
    word = fit_deployable_sklearn(
        build_word_model(), train_x, train_y, calibration_x, calibration_y, selection_x, selection_y
    )
    model_bundle["word_tfidf"] = word
    raw_test_probabilities["word_tfidf"] = word["estimator"].predict_proba(test_x)[:, 1]
    latencies["word_tfidf"] = median_latency(lambda text: word["estimator"].predict_proba([text])[0, 1], test_x)
    sizes["word_tfidf"] = len(pickle.dumps(word)) / (1024 * 1024)

    print("[2/4] Training hybrid word/character TF-IDF model", flush=True)
    hybrid = fit_deployable_sklearn(
        build_hybrid_model(), train_x, train_y, calibration_x, calibration_y, selection_x, selection_y
    )
    model_bundle["hybrid_tfidf"] = hybrid
    raw_test_probabilities["hybrid_tfidf"] = hybrid["estimator"].predict_proba(test_x)[:, 1]
    latencies["hybrid_tfidf"] = median_latency(lambda text: hybrid["estimator"].predict_proba([text])[0, 1], test_x)
    sizes["hybrid_tfidf"] = len(pickle.dumps(hybrid)) / (1024 * 1024)

    history: list[dict[str, float | int]] = []
    bilstm_seed_rows: list[dict[str, float | int]] = []
    if not args.skip_bilstm:
        print("[3/4] Training BiLSTM with early stopping across three fixed seeds", flush=True)
        network = vocab = config = None
        for bilstm_seed in (4158, 4159, 4160):
            candidate_network, candidate_vocab, candidate_config, candidate_history = train_bilstm(
                train_x, train_y, selection_x, selection_y, seed=bilstm_seed
            )
            candidate_probability = bilstm_probabilities(
                candidate_network, selection_x, candidate_vocab, 80
            )
            bilstm_seed_rows.append(
                {
                    "seed": bilstm_seed,
                    "evaluation_split": "selection",
                    "macro_f1_at_0.5": float(
                        f1_score(selection_y, candidate_probability >= 0.5, average="macro")
                    ),
                }
            )
            history.extend(candidate_history)
            if bilstm_seed == SEED:
                network, vocab, config = candidate_network, candidate_vocab, candidate_config
        if network is None or vocab is None or config is None:
            raise RuntimeError("The pre-registered BiLSTM seed did not train")
        bilstm_cal_raw = bilstm_probabilities(network, calibration_x, vocab, 80)
        bilstm_calibrator = fit_platt_calibrator(bilstm_cal_raw, calibration_y)
        bilstm_selection = calibrate(bilstm_probabilities(network, selection_x, vocab, 80), bilstm_calibrator)
        bilstm_threshold = choose_threshold(selection_y, bilstm_selection)
        bilstm_review = choose_review_confidence(selection_y, bilstm_selection, bilstm_threshold)
        state_path = artifacts_dir / "bilstm_state.pt"
        import torch

        torch.save(network.state_dict(), state_path)
        bilstm_artifact = {
            "calibrator": bilstm_calibrator,
            "decision_threshold": bilstm_threshold,
            "review_confidence": bilstm_review,
            "vocab": vocab,
            "config": {
                "vocab_size": config.vocab_size,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "dropout": config.dropout,
            },
            "max_length": 80,
            "state_path": "artifacts/bilstm_state.pt",
        }
        model_bundle["bilstm"] = bilstm_artifact
        raw_test_probabilities["bilstm"] = bilstm_probabilities(network, test_x, vocab, 80)
        latencies["bilstm"] = median_latency(
            lambda text: bilstm_probabilities(network, [text], vocab, 80)[0], test_x
        )
        sizes["bilstm"] = (state_path.stat().st_size + len(pickle.dumps(bilstm_artifact))) / (1024 * 1024)

    if not args.skip_transformer:
        print("[4/4] Encoding with frozen MiniLM and training logistic regression", flush=True)
        from sentence_transformers import SentenceTransformer

        encoder_path = models_dir / "all-MiniLM-L6-v2"
        local_encoder = encoder_path / "model.safetensors"
        encoder_source = str(encoder_path) if local_encoder.exists() else args.transformer
        encoder = SentenceTransformer(encoder_source)
        encoder.save(str(encoder_path))
        embeddings = {
            "train": encoder.encode(train_x, batch_size=64, normalize_embeddings=True, show_progress_bar=True),
            "calibration": encoder.encode(calibration_x, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
            "selection": encoder.encode(selection_x, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
            "test": encoder.encode(test_x, batch_size=64, normalize_embeddings=True, show_progress_bar=False),
        }
        classifier = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2_000, random_state=SEED)
        classifier.fit(embeddings["train"], train_y)
        minilm_cal_raw = classifier.predict_proba(embeddings["calibration"])[:, 1]
        minilm_calibrator = fit_platt_calibrator(minilm_cal_raw, calibration_y)
        minilm_selection = calibrate(classifier.predict_proba(embeddings["selection"])[:, 1], minilm_calibrator)
        minilm_threshold = choose_threshold(selection_y, minilm_selection)
        minilm_review = choose_review_confidence(selection_y, minilm_selection, minilm_threshold)
        minilm_artifact = {
            "classifier": classifier,
            "calibrator": minilm_calibrator,
            "decision_threshold": minilm_threshold,
            "review_confidence": minilm_review,
            "encoder_path": "models/all-MiniLM-L6-v2",
            "encoder_id": args.transformer,
            "training": "frozen sentence embeddings + logistic regression (not fine-tuning)",
        }
        model_bundle["minilm"] = minilm_artifact
        raw_test_probabilities["minilm"] = classifier.predict_proba(embeddings["test"])[:, 1]
        latencies["minilm"] = median_latency(
            lambda text: classifier.predict_proba(
                encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
            )[0, 1],
            test_x,
            runs=10,
        )
        encoder_bytes = sum(path.stat().st_size for path in encoder_path.rglob("*") if path.is_file())
        sizes["minilm"] = (encoder_bytes + len(pickle.dumps(minilm_artifact))) / (1024 * 1024)

    retrieval_vectorizer = hybrid["estimator"].named_steps["features"]
    retrieval_matrix = retrieval_vectorizer.transform(train_x)
    created_utc = datetime.now(timezone.utc).isoformat()
    bundle = {
        "schema_version": 1,
        "created_utc": created_utc,
        "default_model": "hybrid_tfidf",
        "models": model_bundle,
        "dataset": {
            **dedupe_audit,
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "seed": SEED,
        },
        "retrieval": {
            "vectorizer": retrieval_vectorizer,
            "matrix": retrieval_matrix,
            "messages": train["message"].tolist(),
            "labels": train["label_name"].tolist(),
        },
    }
    joblib.dump(bundle, artifacts_dir / "models.joblib", compress=3)
    pd.DataFrame(history).to_csv(results_dir / "bilstm_training_history.csv", index=False)
    pd.DataFrame(bilstm_seed_rows).to_csv(results_dir / "bilstm_seed_results.csv", index=False)

    manifest = {
        "title": "SMS Spam Collection",
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "doi": "10.24432/C5CC84",
        "license": "CC BY 4.0",
        "original_file_sha256": sha256(args.data),
        "raw_rows": int(len(raw)),
        "deduplication": dedupe_audit,
        "split_seed": SEED,
        "split_counts": frame["split"].value_counts().to_dict(),
        "class_counts_after_deduplication": clean["label_name"].value_counts().to_dict(),
        "calibration_protocol": "The 15% validation split is halved: sigmoid calibration on one half; decision and abstention thresholds on the other.",
        "test_policy": "The 15% test split is used once after all choices are frozen.",
        "model_provenance": {
            "minilm_identifier": "sentence-transformers/all-MiniLM-L6-v2",
            "bundled_encoder_sha256": sha256(models_dir / "all-MiniLM-L6-v2" / "model.safetensors") if not args.skip_transformer else None,
            "bilstm_deployment_seed": SEED,
            "bilstm_stability_seeds": [4158, 4159, 4160] if not args.skip_bilstm else [],
        },
    }
    (ROOT / "data" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    labels_by_key = {
        "majority": "Majority-class baseline",
        "word_tfidf": "Word TF-IDF + Logistic Regression",
        "hybrid_tfidf": "Hybrid word/character TF-IDF",
        "bilstm": "BiLSTM",
        "minilm": "MiniLM embeddings + Logistic Regression",
    }
    majority_probability = np.full(len(test_y), float(train_y.mean()))
    calibrated_test: dict[str, np.ndarray] = {"majority": majority_probability}
    metric_rows: list[dict[str, float | str]] = []
    matrices: dict[str, list[list[int]]] = {}
    majority_row, majority_matrix = metric_row(
        "majority", labels_by_key["majority"], test_y, majority_probability, 0.5, 0.5, 0.0, 0.0
    )
    metric_rows.append(majority_row)
    matrices["majority"] = majority_matrix
    for key, raw_probability in raw_test_probabilities.items():
        model = model_bundle[key]
        probability = calibrate(raw_probability, model["calibrator"])
        calibrated_test[key] = probability
        row, matrix = metric_row(
            key,
            labels_by_key[key],
            test_y,
            probability,
            float(model["decision_threshold"]),
            float(model["review_confidence"]),
            latencies[key],
            sizes[key],
        )
        metric_rows.append(row)
        matrices[key] = matrix
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "model_metrics.csv", index=False)
    (results_dir / "confusion_matrices.json").write_text(json.dumps(matrices, indent=2), encoding="utf-8")

    prediction_frame = test[["message_id", "message", "label", "label_name"]].copy()
    for key, probability in calibrated_test.items():
        threshold = 0.5 if key == "majority" else float(model_bundle[key]["decision_threshold"])
        review = 0.5 if key == "majority" else float(model_bundle[key]["review_confidence"])
        prediction_frame[f"{key}_probability"] = probability
        prediction_frame[f"{key}_prediction"] = (probability >= threshold).astype(int)
        prediction_frame[f"{key}_review"] = np.maximum(probability, 1 - probability) < review
    prediction_frame.to_csv(results_dir / "test_predictions.csv", index=False)

    print("Running validation-only ablations", flush=True)
    ablations = [
        evaluate_ablation(
            "word unigrams, normalized",
            build_word_model(ngram_range=(1, 1)),
            train_x,
            train_y,
            selection_x,
            selection_y,
        ),
        evaluate_ablation(
            "word unigrams+bigrams, normalized",
            build_word_model(ngram_range=(1, 2)),
            train_x,
            train_y,
            selection_x,
            selection_y,
        ),
        evaluate_ablation(
            "word unigrams+bigrams, raw surface text",
            build_word_model(ngram_range=(1, 2), normalized=False),
            train["message"].tolist(),
            train_y,
            selection_frame["message"].tolist(),
            selection_y,
        ),
        evaluate_ablation(
            "word+character hybrid, normalized",
            build_hybrid_model(),
            train_x,
            train_y,
            selection_x,
            selection_y,
        ),
    ]
    pd.DataFrame(ablations).to_csv(results_dir / "ablation_results.csv", index=False)
    print(metrics[["model", "macro_f1", "spam_recall", "pr_auc", "brier_score"]].to_string(index=False))
    print(f"Artifacts written to {artifacts_dir}", flush=True)


if __name__ == "__main__":
    main()
