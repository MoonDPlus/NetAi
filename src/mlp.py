from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6 / (in_features + out_features))
        self.w = rng.uniform(-limit, limit, size=(in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.x: np.ndarray | None = None
        self.dw: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Forward must be called before backward.")
        m = self.x.shape[0]
        self.dw = self.x.T @ grad_output / m
        self.db = np.sum(grad_output, axis=0, keepdims=True) / m
        return grad_output @ self.w.T

    def step(self, lr: float) -> None:
        if self.dw is None or self.db is None:
            raise RuntimeError("Backward must be called before step.")
        self.w -= lr * self.dw
        self.b -= lr * self.db


class ReLU:
    def __init__(self) -> None:
        self.x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Forward must be called before backward.")
        return grad_output * (self.x > 0.0)


class Sigmoid:
    def __init__(self) -> None:
        self.y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self.y is None:
            raise RuntimeError("Forward must be called before backward.")
        return grad_output * self.y * (1.0 - self.y)


class BinaryCrossEntropy:
    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps
        self.pred: np.ndarray | None = None
        self.target: np.ndarray | None = None

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = np.clip(pred, self.eps, 1.0 - self.eps)
        self.target = target
        loss = -np.mean(
            target * np.log(self.pred) + (1.0 - target) * np.log(1.0 - self.pred)
        )
        return float(loss)

    def backward(self) -> np.ndarray:
        if self.pred is None or self.target is None:
            raise RuntimeError("Forward must be called before backward.")
        return (self.pred - self.target) / (self.pred * (1.0 - self.pred))


class MLPBinaryClassifier:
    def __init__(self, seed: int = 42) -> None:
        self.fc1 = Dense(2, 8, seed=seed)
        self.relu = ReLU()
        self.fc2 = Dense(8, 1, seed=seed + 1)
        self.sigmoid = Sigmoid()

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = self.fc1.forward(x)
        a1 = self.relu.forward(z1)
        z2 = self.fc2.forward(a1)
        return self.sigmoid.forward(z2)

    def backward(self, grad_output: np.ndarray) -> None:
        grad = self.sigmoid.backward(grad_output)
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        self.fc1.backward(grad)

    def step(self, lr: float) -> None:
        self.fc1.step(lr)
        self.fc2.step(lr)

    def save(self, path: str | Path) -> None:
        payload = {
            "fc1_w": self.fc1.w.tolist(),
            "fc1_b": self.fc1.b.tolist(),
            "fc2_w": self.fc2.w.tolist(),
            "fc2_b": self.fc2.b.tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.fc1.w = np.array(payload["fc1_w"], dtype=float)
        self.fc1.b = np.array(payload["fc1_b"], dtype=float)
        self.fc2.w = np.array(payload["fc2_w"], dtype=float)
        self.fc2.b = np.array(payload["fc2_b"], dtype=float)
