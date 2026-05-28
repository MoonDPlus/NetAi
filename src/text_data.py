from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_labeled_text_csv(path: str | Path) -> tuple[list[str], list[int]]:
    """Load `text,label` rows from CSV file."""
    texts: list[str] = []
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            label = int((row.get("label") or "0").strip())
            if text:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise ValueError("No data loaded from CSV")
    return texts, labels


def train_val_test_split_text(
    texts: list[str],
    labels: list[int],
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    """Split text dataset into train/val/test partitions."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have same length")
    if len(texts) < 3:
        raise ValueError("need at least 3 samples")
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio and test_ratio must be >0 and sum to <1")

    n = len(texts)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("not enough samples for train split")

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def pick(indices: np.ndarray) -> tuple[list[str], list[int]]:
        return [texts[i] for i in indices], [labels[i] for i in indices]

    x_train, y_train = pick(train_idx)
    x_val, y_val = pick(val_idx)
    x_test, y_test = pick(test_idx)
    return x_train, y_train, x_val, y_val, x_test, y_test
