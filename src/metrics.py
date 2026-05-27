from __future__ import annotations

import numpy as np


def confusion_counts(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[int, int, int, int]:
    p = (y_pred >= 0.5).astype(int).reshape(-1)
    y = y_true.astype(int).reshape(-1)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def classification_report(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float | dict[str, int]]:
    tp, tn, fp, fn = confusion_counts(y_pred, y_true)
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
