#!/usr/bin/env python3
"""Audit the frozen evaluation outputs without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    metrics_path = ROOT / "results" / "model_metrics.csv"
    predictions_path = ROOT / "results" / "test_predictions.csv"
    artifact_path = ROOT / "artifacts" / "models.joblib"
    missing = [path for path in (metrics_path, predictions_path, artifact_path) if not path.exists()]
    if missing:
        raise SystemExit("Missing files; run train.py first: " + ", ".join(str(path) for path in missing))
    bundle = joblib.load(artifact_path)
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    evaluated = set(metrics["model_key"]) - {"majority"}
    packaged = set(bundle["models"])
    if evaluated != packaged:
        raise SystemExit(f"Artifact/metric mismatch: evaluated={evaluated}, packaged={packaged}")
    if len(predictions) != manifest["split_counts"]["test"]:
        raise SystemExit("Prediction count does not match the frozen test split")
    print("SpamScope frozen evaluation audit: PASS")
    print(f"Dataset: {manifest['title']} | SHA-256 {manifest['original_file_sha256']}")
    print(f"Test examples: {len(predictions)} | Models: {len(metrics)}")
    print(metrics[["model", "macro_f1", "spam_precision", "spam_recall", "pr_auc", "brier_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
