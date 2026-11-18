from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import deduplicate, load_uci_sms, normalize_text, stratified_split, tokenize


class PreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_uci_sms(ROOT / "data" / "SMSSpamCollection")
        cls.clean, cls.audit = deduplicate(cls.raw)
        cls.split = stratified_split(cls.clean)

    def test_official_source_row_count(self):
        self.assertEqual(len(self.raw), 5_574)

    def test_label_mapping_is_explicit(self):
        mapping = self.raw.groupby("label_name")["label"].unique().to_dict()
        self.assertEqual(mapping["ham"].tolist(), [0])
        self.assertEqual(mapping["spam"].tolist(), [1])

    def test_normalization_masks_transport_identifiers(self):
        text = "Email A@b.com or visit https://x.test; call +44 7700 900123 for £50"
        normalized = normalize_text(text)
        self.assertIn("<EMAIL>", normalized)
        self.assertIn("<URL>", normalized)
        self.assertIn("<PHONE>", normalized)
        self.assertIn("<MONEY>", normalized)

    def test_normalization_retains_spam_wording(self):
        normalized = normalize_text("URGENT! Claim your FREE prize now")
        self.assertEqual(normalized, "urgent! claim your free prize now")

    def test_tokenizer_is_deterministic_and_nonempty(self):
        first = tokenize("Win £50 NOW!!!")
        self.assertEqual(first, tokenize("Win £50 NOW!!!"))
        self.assertGreater(len(first), 0)

    def test_deduplication_happens_before_split(self):
        self.assertEqual(self.audit["duplicate_rows_removed"], 429)
        self.assertEqual(len(self.clean), 5_145)
        self.assertTrue(self.clean["normalized_text"].is_unique)

    def test_split_counts_are_fixed(self):
        self.assertEqual(
            self.split["split"].value_counts().to_dict(),
            {"train": 3_601, "test": 772, "validation": 772},
        )

    def test_no_message_crosses_splits(self):
        groups = self.split.groupby("message_id")["split"].nunique()
        self.assertEqual(int((groups > 1).sum()), 0)

    def test_stratification_preserves_prevalence(self):
        rates = self.split.groupby("split")["label"].mean()
        self.assertLess(float(rates.max() - rates.min()), 0.002)

    def test_split_is_deterministic(self):
        rerun = stratified_split(self.clean)
        left = self.split.sort_values("message_id").set_index("message_id")["split"]
        right = rerun.sort_values("message_id").set_index("message_id")["split"]
        self.assertTrue(left.equals(right))


if __name__ == "__main__":
    unittest.main()
