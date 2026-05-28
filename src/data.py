from __future__ import annotations

import numpy as np


def make_xor_dataset(repeats: int = 1, noise: float = 0.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return XOR dataset, optionally repeated with Gaussian noise."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    base_x = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    base_y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)

    x = np.tile(base_x, (repeats, 1))
    y = np.tile(base_y, (repeats, 1))

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        x = x + rng.normal(0.0, noise, size=x.shape)

    return x, y


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/validation subsets."""
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    n = len(x)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    val_count = max(1, int(n * val_ratio))

    val_idx = indices[:val_count]
    train_idx = indices[val_count:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def iter_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
):
    """Yield mini-batches from dataset."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    n = len(x)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]
