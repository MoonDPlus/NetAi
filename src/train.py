from __future__ import annotations

import argparse

import numpy as np

from src.data import iter_minibatches, make_xor_dataset, train_val_split
from src.mlp import BinaryCrossEntropy, MLPBinaryClassifier


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    pred_labels = (y_pred >= 0.5).astype(float)
    return float(np.mean(pred_labels == y_true))


def train(
    epochs: int = 2000,
    lr: float = 0.1,
    seed: int = 42,
    batch_size: int = 16,
    repeats: int = 200,
    noise: float = 0.05,
    save_path: str | None = None,
    optimizer: str = "sgd",
    patience: int = 200,
) -> None:
    x, y = make_xor_dataset(repeats=repeats, noise=noise, seed=seed)
    x_train, y_train, x_val, y_val = train_val_split(x, y, val_ratio=0.2, seed=seed)

    model = MLPBinaryClassifier(seed=seed)
    criterion = BinaryCrossEntropy()

    best_val_loss = float("inf")
    best_epoch = 0
    steps_total = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0

        for xb, yb in iter_minibatches(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            y_pred = model.forward(xb)
            loss = criterion.forward(y_pred, yb)
            grad = criterion.backward()
            model.backward(grad)
            steps_total += 1
            if optimizer == "adam":
                model.step_adam(lr=lr, t=steps_total)
            else:
                model.step(lr)

            epoch_loss += loss
            steps += 1

        train_pred = model.forward(x_train)
        val_pred = model.forward(x_val)
        train_acc = accuracy(train_pred, y_train)
        val_acc = accuracy(val_pred, y_val)
        avg_loss = epoch_loss / max(1, steps)
        val_loss = criterion.forward(val_pred, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if save_path:
                model.save(save_path)
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
            break

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d} loss={avg_loss:.6f} val_loss={val_loss:.6f} "
                f"train_acc={train_acc:.2%} val_acc={val_acc:.2%}"
            )

    if save_path:
        print(f"Best model saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP from scratch on XOR dataset")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--patience", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
        repeats=args.repeats,
        noise=args.noise,
        save_path=args.save_path or None,
        optimizer=args.optimizer,
        patience=args.patience,
    )
