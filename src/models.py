"""Model builders, calibration utilities, and the compact BiLSTM."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .preprocess import tokenize


def seed_everything(seed: int = 4158) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def build_word_model(*, ngram_range: tuple[int, int] = (1, 2), normalized: bool = True) -> Pipeline:
    """Interpretable word TF-IDF logistic-regression pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=normalized,
    )
    classifier = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=4158,
        solver="liblinear",
    )
    return Pipeline([("features", vectorizer), ("classifier", classifier)])


def build_hybrid_model() -> Pipeline:
    """Combine word semantics with character patterns that handle noisy SMS."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=25_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=35_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=3.0,
        class_weight="balanced",
        max_iter=2_000,
        random_state=4158,
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt_calibrator(probabilities: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1_000)
    calibrator.fit(probabilities_to_logits(probabilities), labels)
    return calibrator


def calibrate(probabilities: np.ndarray, calibrator: LogisticRegression) -> np.ndarray:
    return calibrator.predict_proba(probabilities_to_logits(probabilities))[:, 1]


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    best = (float("-inf"), 0.5)
    for threshold in np.linspace(0.20, 0.80, 121):
        score = f1_score(labels, probabilities >= threshold, average="macro")
        candidate = (float(score), -abs(float(threshold) - 0.5), float(threshold))
        if candidate[:2] > (best[0], -abs(best[1] - 0.5)):
            best = (candidate[0], candidate[2])
    return round(best[1], 3)


def choose_review_confidence(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    minimum_coverage: float = 0.80,
) -> float:
    """Select abstention cut-off on validation data only."""
    confidence = np.maximum(probabilities, 1 - probabilities)
    predictions = probabilities >= threshold
    best: tuple[float, float, float] | None = None
    for candidate in np.linspace(0.50, 0.95, 91):
        retained = confidence >= candidate
        coverage = float(retained.mean())
        if coverage < minimum_coverage or not retained.any():
            continue
        accuracy = float((predictions[retained] == labels[retained]).mean())
        key = (accuracy, float(candidate), coverage)
        if best is None or key > best:
            best = key
    return round(best[1] if best else 0.5, 3)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0, 1, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(value if total else math.nan)


def build_vocabulary(texts: list[str], *, min_frequency: int = 2, max_size: int = 12_000) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in ordered:
        if count < min_frequency or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], *, max_length: int = 80) -> list[int]:
    encoded = [vocab.get(token, 1) for token in tokenize(text)][:max_length]
    return encoded or [1]


@dataclass(frozen=True)
class BiLSTMConfig:
    vocab_size: int
    embedding_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.25


def make_bilstm(config: BiLSTMConfig):
    """Construct lazily so classical-only workflows do not require PyTorch."""
    import torch
    from torch import nn

    class BiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)
            self.encoder = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(config.hidden_dim * 2, 1)

        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.encoder(packed)
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.output(self.dropout(features)).squeeze(1)

    return BiLSTM()
