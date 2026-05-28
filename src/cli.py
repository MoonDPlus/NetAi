from __future__ import annotations

import argparse
import json

import numpy as np

from src.features import BagOfWordsVectorizer

from src.dataset_collect import collect_from_sitemaps, collect_from_urls
from src.metrics import classification_report
from src.corpus_tools import load_texts_from_csv, save_stats_json, stats_from_csv
from src.generate import generate_sentences
from src.mlp import BinaryCrossEntropy, MLPBinaryClassifier
from src.text_data import load_labeled_text_csv, train_val_test_split_text
from src.chatbot import MemoryChatbot


def _to_xy(texts: list[str], labels: list[int], vec: BagOfWordsVectorizer) -> tuple[np.ndarray, np.ndarray]:
    x = vec.transform(texts)
    y = np.array(labels, dtype=float).reshape(-1, 1)
    return x, y


def _train_text(args: argparse.Namespace) -> None:
    texts, labels = load_labeled_text_csv(args.data)
    x_train_t, y_train_l, x_val_t, y_val_l, x_test_t, y_test_l = train_val_test_split_text(
        texts, labels, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    vec = BagOfWordsVectorizer(max_features=args.max_features)
    vec.fit(x_train_t)

    x_train, y_train = _to_xy(x_train_t, y_train_l, vec)
    x_val, y_val = _to_xy(x_val_t, y_val_l, vec)
    x_test, y_test = _to_xy(x_test_t, y_test_l, vec)

    model = MLPBinaryClassifier(seed=args.seed, in_features=x_train.shape[1], hidden_size=args.hidden_size)
    criterion = BinaryCrossEntropy()

    best_val = float("inf")
    best_epoch = 0
    steps = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        pred = model.forward(x_train)
        train_loss = criterion.forward(pred, y_train)
        grad = criterion.backward()
        model.backward(grad)

        steps += 1
        if args.optimizer == "adam":
            model.step_adam(args.lr, steps)
        else:
            model.step(args.lr)

        val_pred = model.forward(x_val)
        val_loss = criterion.forward(val_pred, y_val)
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            model.save(args.model_path)
            vec.save(args.vocab_path)
        elif epoch - best_epoch >= args.patience:
            break

    best_model = MLPBinaryClassifier(seed=args.seed, in_features=x_train.shape[1], hidden_size=args.hidden_size)
    best_model.load(args.model_path)
    test_pred = best_model.forward(x_test)
    test_report = classification_report(test_pred, y_test)

    if args.history_path:
        with open(args.history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_report": test_report,
        "model_path": args.model_path,
        "vocab_path": args.vocab_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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




def _collect_text(args: argparse.Namespace) -> None:
    if args.mode == "urls":
        with open(args.input, "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]
        count = collect_from_urls(urls, args.out_csv, min_chars=args.min_chars, delay_sec=args.delay_sec, verbose=args.verbose)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            sitemaps = [line.strip() for line in f if line.strip()]
        count = collect_from_sitemaps(sitemaps, args.out_csv, max_urls=args.max_urls, verbose=args.verbose)
    print(json.dumps({"saved": args.out_csv, "rows": count}, ensure_ascii=False, indent=2))



def _analyze_corpus(args: argparse.Namespace) -> None:
    stats = stats_from_csv(args.data_csv)
    if args.out_json:
        save_stats_json(stats, args.out_json)
    print(json.dumps(stats, ensure_ascii=False, indent=2))



def _generate_text(args: argparse.Namespace) -> None:
    texts = load_texts_from_csv(args.data_csv)
    generated = generate_sentences(texts, n_sentences=args.n_sentences, max_words=args.max_words, seed=args.seed)
    print(json.dumps({"count": len(generated), "sentences": generated}, ensure_ascii=False, indent=2))



def _chat_reply(args: argparse.Namespace) -> None:
    bot = MemoryChatbot(memory_path=args.memory_path)
    result = bot.respond(args.message)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _chat_learn(args: argparse.Namespace) -> None:
    bot = MemoryChatbot(memory_path=args.memory_path)
    bot.learn_pair(args.user_message, args.assistant_message)
    print(json.dumps({"saved": True, "memory_path": args.memory_path, "items": len(bot.pairs)}, ensure_ascii=False, indent=2))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified CLI for text train/eval")
    sp = p.add_subparsers(dest="cmd", required=True)

    train_p = sp.add_parser("train-text")
    train_p.add_argument("--data", required=True)
    train_p.add_argument("--model-path", default="text_model.json")
    train_p.add_argument("--vocab-path", default="vocab.json")
    train_p.add_argument("--history-path", default="text_history.json")
    train_p.add_argument("--epochs", type=int, default=200)
    train_p.add_argument("--lr", type=float, default=0.05)
    train_p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    train_p.add_argument("--max-features", type=int, default=2000)
    train_p.add_argument("--hidden-size", type=int, default=16)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--val-ratio", type=float, default=0.2)
    train_p.add_argument("--test-ratio", type=float, default=0.2)
    train_p.add_argument("--patience", type=int, default=50)
    train_p.set_defaults(func=_train_text)

    eval_p = sp.add_parser("eval-text")
    eval_p.add_argument("--data", required=True)
    eval_p.add_argument("--model-path", default="text_model.json")
    eval_p.add_argument("--vocab-path", default="vocab.json")
    eval_p.add_argument("--hidden-size", type=int, default=16)
    eval_p.set_defaults(func=_eval_text)

    collect_p = sp.add_parser("collect-text")
    collect_p.add_argument("--mode", choices=["urls", "sitemaps"], default="urls")
    collect_p.add_argument("--input", required=True, help="Text file containing one URL per line")
    collect_p.add_argument("--out-csv", default="data/raw_web_text.csv")
    collect_p.add_argument("--min-chars", type=int, default=200)
    collect_p.add_argument("--delay-sec", type=float, default=1.0)
    collect_p.add_argument("--max-urls", type=int, default=200)
    collect_p.add_argument("--verbose", action="store_true")
    collect_p.set_defaults(func=_collect_text)

    analyze_p = sp.add_parser("analyze-corpus")
    analyze_p.add_argument("--data-csv", required=True)
    analyze_p.add_argument("--out-json", default="")
    analyze_p.set_defaults(func=_analyze_corpus)

    gen_p = sp.add_parser("generate-text")
    gen_p.add_argument("--data-csv", required=True)
    gen_p.add_argument("--n-sentences", type=int, default=5)
    gen_p.add_argument("--max-words", type=int, default=12)
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.set_defaults(func=_generate_text)

    chat_p = sp.add_parser("chat")
    chat_p.add_argument("--message", required=True)
    chat_p.add_argument("--memory-path", default="data/chat_memory.json")
    chat_p.set_defaults(func=_chat_reply)

    learn_p = sp.add_parser("learn-chat")
    learn_p.add_argument("--user-message", required=True)
    learn_p.add_argument("--assistant-message", required=True)
    learn_p.add_argument("--memory-path", default="data/chat_memory.json")
    learn_p.set_defaults(func=_chat_learn)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
