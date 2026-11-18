"""Dataset loading, leakage-safe splitting, and text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
MONEY_RE = re.compile(r"(?:[$£€]\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?(?:pounds?|dollars?|euros?)\b)", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"<[^>]+>|[a-z0-9]+(?:'[a-z]+)?|[!?£$€]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize transport noise while retaining spam-relevant wording."""
    value = unicodedata.normalize("NFKC", str(text)).strip().lower()
    value = URL_RE.sub(" <URL> ", value)
    value = EMAIL_RE.sub(" <EMAIL> ", value)
    value = MONEY_RE.sub(" <MONEY> ", value)
    value = PHONE_RE.sub(" <PHONE> ", value)
    return SPACE_RE.sub(" ", value).strip()


def tokenize(text: str) -> list[str]:
    """Small deterministic tokenizer used by the BiLSTM."""
    return TOKEN_RE.findall(normalize_text(text))


def _message_id(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def load_uci_sms(path: str | Path) -> pd.DataFrame:
    """Load the original UCI tab-separated file and validate every row."""
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                label_name, message = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed dataset row {line_no}") from exc
            if label_name not in {"ham", "spam"}:
                raise ValueError(f"Unexpected label {label_name!r} on row {line_no}")
            normalized = normalize_text(message)
            rows.append(
                {
                    "source_row": line_no,
                    "message": message,
                    "normalized_text": normalized,
                    "label": int(label_name == "spam"),
                    "label_name": label_name,
                    "message_id": _message_id(normalized),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("The dataset is empty")
    return frame


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove normalized duplicates before splitting to prevent leakage."""
    required = {"normalized_text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    label_counts = frame.groupby("normalized_text")["label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    without_conflicts = frame[~frame["normalized_text"].isin(conflicts)].copy()
    clean = without_conflicts.drop_duplicates("normalized_text", keep="first").reset_index(drop=True)
    audit = {
        "raw_rows": int(len(frame)),
        "conflicting_texts_removed": int(len(conflicts)),
        "duplicate_rows_removed": int(len(without_conflicts) - len(clean)),
        "clean_rows": int(len(clean)),
    }
    return clean, audit


def stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int = 4158,
    train_size: float = 0.70,
) -> pd.DataFrame:
    """Create a deterministic 70/15/15 stratified split after deduplication."""
    if not 0.5 <= train_size < 1:
        raise ValueError("train_size must be between 0.5 and 1.0")
    train_idx, holdout_idx = train_test_split(
        frame.index,
        train_size=train_size,
        random_state=seed,
        stratify=frame["label"],
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.5,
        random_state=seed,
        stratify=frame.loc[holdout_idx, "label"],
    )
    result = frame.copy()
    result["split"] = ""
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "validation"
    result.loc[test_idx, "split"] = "test"
    if (result["split"] == "").any():
        raise RuntimeError("Some rows were not assigned to a split")
    return result.reset_index(drop=True)
