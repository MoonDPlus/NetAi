from __future__ import annotations

import argparse

import numpy as np

from src.mlp import MLPBinaryClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with saved XOR MLP model")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--x1", type=float, required=True)
    parser.add_argument("--x2", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = MLPBinaryClassifier(seed=42)
    model.load(args.model_path)

    x = np.array([[args.x1, args.x2]], dtype=float)
    p = model.forward(x).item()
    label = 1 if p >= 0.5 else 0
    print(f"input={[args.x1, args.x2]} prob={p:.6f} label={label}")
