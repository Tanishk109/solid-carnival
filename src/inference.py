"""Inference API shared by Streamlit and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import BiLSTMConfig, calibrate, encode_text, make_bilstm
from .preprocess import normalize_text


MODEL_LABELS = {
    "word_tfidf": "Word TF-IDF + Logistic Regression",
    "hybrid_tfidf": "Hybrid word/character TF-IDF",
    "bilstm": "BiLSTM",
    "minilm": "MiniLM embeddings + Logistic Regression",
}

DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SafeTextEngine:
    """Load the exact evaluated artifacts and expose stable prediction methods."""

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)
        artifact_path = self.project_dir / "artifacts" / "models.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing trained artifact: {artifact_path}")
        self.bundle: dict[str, Any] = joblib.load(artifact_path)
        self._embedder = None
        self._bilstm = None

    @property
    def available_models(self) -> list[str]:
        return [key for key in MODEL_LABELS if key in self.bundle["models"]]

    def _raw_probability(self, normalized: str, model_key: str) -> float:
        model = self.bundle["models"][model_key]
        if model_key in {"word_tfidf", "hybrid_tfidf"}:
            return float(model["estimator"].predict_proba([normalized])[0, 1])
        if model_key == "minilm":
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                local_encoder = self.project_dir / model["encoder_path"]
                encoder = str(local_encoder) if local_encoder.exists() else DEFAULT_MINILM_MODEL
                self._embedder = SentenceTransformer(encoder)
            vector = self._embedder.encode([normalized], normalize_embeddings=True, show_progress_bar=False)
            return float(model["classifier"].predict_proba(vector)[0, 1])
        if model_key == "bilstm":
            import torch

            if self._bilstm is None:
                config = BiLSTMConfig(**model["config"])
                network = make_bilstm(config)
                state = torch.load(self.project_dir / model["state_path"], map_location="cpu", weights_only=True)
                network.load_state_dict(state)
                network.eval()
                self._bilstm = network
            sequence = encode_text(normalized, model["vocab"], max_length=model["max_length"])
            tokens = torch.tensor([sequence], dtype=torch.long)
            lengths = torch.tensor([len(sequence)], dtype=torch.long)
            with torch.inference_mode():
                return float(torch.sigmoid(self._bilstm(tokens, lengths)).item())
        raise KeyError(f"Unknown model: {model_key}")

    def predict(self, text: str, model_key: str = "hybrid_tfidf") -> dict[str, Any]:
        if model_key not in self.available_models:
            raise KeyError(f"Unavailable model: {model_key}")
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("Enter a non-empty message")
        model = self.bundle["models"][model_key]
        raw_probability = self._raw_probability(normalized, model_key)
        probability = float(calibrate(np.array([raw_probability]), model["calibrator"])[0])
        threshold = float(model["decision_threshold"])
        confidence = float(max(probability, 1 - probability))
        needs_review = confidence < float(model["review_confidence"])
        predicted_label = int(probability >= threshold)
        return {
            "model_key": model_key,
            "model_name": MODEL_LABELS[model_key],
            "label": predicted_label,
            "label_name": "spam" if predicted_label else "ham",
            "spam_probability": probability,
            "confidence": confidence,
            "decision_threshold": threshold,
            "review_confidence": float(model["review_confidence"]),
            "needs_review": bool(needs_review),
            "normalized_text": normalized,
        }

    def explain_linear(self, text: str, model_key: str) -> dict[str, Any]:
        if model_key not in {"word_tfidf", "hybrid_tfidf"}:
            return {
                "supported": False,
                "note": "This model has no faithful token coefficients. Similar examples are shown instead.",
            }
        estimator = self.bundle["models"][model_key]["estimator"]
        features = estimator.named_steps["features"]
        classifier = estimator.named_steps["classifier"]
        vector = features.transform([normalize_text(text)]).tocsr()
        names = features.get_feature_names_out()
        values = vector.data * classifier.coef_[0, vector.indices]
        rows = [
            {"feature": str(names[index]), "contribution": float(value)}
            for index, value in zip(vector.indices, values)
        ]
        positive = sorted((row for row in rows if row["contribution"] > 0), key=lambda row: -row["contribution"])[:8]
        negative = sorted((row for row in rows if row["contribution"] < 0), key=lambda row: row["contribution"])[:8]
        return {"supported": True, "toward_spam": positive, "toward_ham": negative}

    def nearest_examples(self, text: str, *, limit: int = 3) -> list[dict[str, Any]]:
        retrieval = self.bundle["retrieval"]
        query = retrieval["vectorizer"].transform([normalize_text(text)])
        scores = cosine_similarity(query, retrieval["matrix"]).ravel()
        order = np.argsort(scores)[::-1][:limit]
        return [
            {
                "message": retrieval["messages"][int(index)],
                "label_name": retrieval["labels"][int(index)],
                "similarity": float(scores[int(index)]),
            }
            for index in order
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "created_utc": self.bundle["created_utc"],
            "dataset": self.bundle["dataset"],
            "default_model": self.bundle["default_model"],
            "models": {key: MODEL_LABELS[key] for key in self.available_models},
        }


def load_manifest(project_dir: str | Path) -> dict[str, Any]:
    path = Path(project_dir) / "data" / "dataset_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
