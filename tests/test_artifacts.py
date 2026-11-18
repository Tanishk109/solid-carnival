from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import DEFAULT_MINILM_MODEL, SafeTextEngine


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SafeTextEngine(ROOT)
        cls.bundle = joblib.load(ROOT / "artifacts" / "models.joblib")
        cls.metrics = pd.read_csv(ROOT / "results" / "model_metrics.csv")
        cls.predictions = pd.read_csv(ROOT / "results" / "test_predictions.csv")
        cls.splits = pd.read_csv(ROOT / "data" / "splits.csv")
        cls.manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))

    def test_all_reported_deployable_models_are_packaged(self):
        reported = set(self.metrics["model_key"]) - {"majority"}
        self.assertEqual(reported, set(self.bundle["models"]))
        self.assertEqual(reported, {"word_tfidf", "hybrid_tfidf", "bilstm", "minilm"})

    def test_metrics_use_entire_test_split(self):
        self.assertTrue((self.metrics["test_examples"] == 772).all())
        self.assertEqual(len(self.predictions), 772)

    def test_prediction_probabilities_are_bounded(self):
        columns = [column for column in self.predictions if column.endswith("_probability")]
        self.assertGreaterEqual(len(columns), 5)
        for column in columns:
            self.assertTrue(self.predictions[column].between(0, 1).all(), column)

    def test_test_ids_match_frozen_split(self):
        expected = set(self.splits.loc[self.splits["split"] == "test", "message_id"])
        self.assertEqual(set(self.predictions["message_id"]), expected)

    def test_thresholds_are_model_specific_and_valid(self):
        thresholds = []
        for model in self.bundle["models"].values():
            thresholds.append(float(model["decision_threshold"]))
            self.assertGreaterEqual(float(model["review_confidence"]), 0.5)
            self.assertLessEqual(float(model["review_confidence"]), 1.0)
        self.assertGreater(len(set(thresholds)), 1)
        self.assertTrue(all(0 < value < 1 for value in thresholds))

    def test_obvious_spam_is_identified(self):
        result = self.engine.predict("URGENT! You won £1000. Call 09061701461 now to claim.", "hybrid_tfidf")
        self.assertEqual(result["label_name"], "spam")
        self.assertGreater(result["spam_probability"], 0.9)

    def test_normal_conversation_is_not_identified_as_spam(self):
        result = self.engine.predict("I will reach the library around six. Please save me a seat.", "hybrid_tfidf")
        self.assertEqual(result["label_name"], "ham")
        self.assertLess(result["spam_probability"], 0.2)

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.predict("   ", "hybrid_tfidf")

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(KeyError):
            self.engine.predict("hello", "imaginary_model")

    def test_linear_explanation_returns_active_features(self):
        result = self.engine.explain_linear("Free prize! Call now to claim cash", "hybrid_tfidf")
        self.assertTrue(result["supported"])
        self.assertGreater(len(result["toward_spam"]), 0)

    def test_transformer_explanation_does_not_claim_token_faithfulness(self):
        result = self.engine.explain_linear("Free prize", "minilm")
        self.assertFalse(result["supported"])
        self.assertIn("no faithful token coefficients", result["note"])

    def test_nearest_examples_are_limited_and_labelled(self):
        examples = self.engine.nearest_examples("claim your free cash prize", limit=3)
        self.assertEqual(len(examples), 3)
        self.assertTrue(all(row["label_name"] in {"ham", "spam"} for row in examples))
        self.assertTrue(all(0 <= row["similarity"] <= 1 for row in examples))

    def test_transformer_uses_reproducible_identifier(self):
        self.assertEqual(DEFAULT_MINILM_MODEL, "sentence-transformers/all-MiniLM-L6-v2")

    def test_bilstm_stability_has_three_preregistered_seeds(self):
        seeds = pd.read_csv(ROOT / "results" / "bilstm_seed_results.csv")
        self.assertEqual(seeds["seed"].tolist(), [4158, 4159, 4160])
        self.assertTrue(seeds["macro_f1_at_0.5"].between(0, 1).all())

    def test_error_analysis_contains_manual_failure_taxonomy(self):
        errors = pd.read_csv(ROOT / "results" / "error_analysis.csv")
        self.assertGreaterEqual(len(errors), 25)
        self.assertGreaterEqual(errors["category"].nunique(), 5)
        self.assertFalse(errors.isna().any().any())

    def test_hybrid_beats_majority_macro_f1(self):
        scores = self.metrics.set_index("model_key")["macro_f1"]
        self.assertGreater(scores["hybrid_tfidf"], scores["majority"] + 0.4)

    def test_ablation_is_not_run_on_test_split(self):
        ablation = pd.read_csv(ROOT / "results" / "ablation_results.csv")
        self.assertEqual(set(ablation["evaluation_split"]), {"selection"})


if __name__ == "__main__":
    unittest.main()
