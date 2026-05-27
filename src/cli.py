from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.features import BagOfWordsVectorizer
from src.metrics import classification_report
from src.mlp import BinaryCrossEntropy, MLPBinaryClassifier
from src.text_data import load_labeled_text_csv


def _train_text(args: argparse.Namespace) -> None:
    texts, labels = load_labeled_text_csv(args.data)
    vec = BagOfWordsVectorizer(max_features=args.max_features)
    x = vec.fit_transform(texts)
    y = np.array(labels, dtype=float).reshape(-1, 1)

    model = MLPBinaryClassifier(seed=args.seed, in_features=x.shape[1], hidden_size=args.hidden_size)
    criterion = BinaryCrossEntropy()

    for _ in range(args.epochs):
        pred = model.forward(x)
        _ = criterion.forward(pred, y)
        grad = criterion.backward()
        model.backward(grad)
        if args.optimizer == "adam":
            model.step_adam(args.lr, 1)
        else:
            model.step(args.lr)

    model.save(args.model_path)
    vec.save(args.vocab_path)
    print(f"saved model={args.model_path} vocab={args.vocab_path}")


def _eval_text(args: argparse.Namespace) -> None:
    texts, labels = load_labeled_text_csv(args.data)
    vec = BagOfWordsVectorizer.load(args.vocab_path)
    x = vec.transform(texts)
    y = np.array(labels, dtype=float).reshape(-1, 1)

    model = MLPBinaryClassifier(seed=42, in_features=x.shape[1], hidden_size=args.hidden_size)
    model.load(args.model_path)
    pred = model.forward(x)
    report = classification_report(pred, y)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for text train/eval")
    sp = p.add_subparsers(dest="cmd", required=True)

    train_p = sp.add_parser("train-text")
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--model-path", default="text_model.json")
    train_p.add_argument("--vocab-path", default="vocab.json")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=0.05)
    train_p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    train_p.add_argument("--max-features", type=int, default=2000)
    train_p.add_argument("--hidden-size", type=int, default=16)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.set_defaults(func=_train_text)

    eval_p = sp.add_parser("eval-text")
    eval_p.add_argument("--data", required=True)
    eval_p.add_argument("--model-path", default="text_model.json")
    eval_p.add_argument("--vocab-path", default="vocab.json")
    eval_p.add_argument("--hidden-size", type=int, default=16)
    eval_p.set_defaults(func=_eval_text)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
