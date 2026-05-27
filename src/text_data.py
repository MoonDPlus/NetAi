from __future__ import annotations

import csv
from pathlib import Path


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
