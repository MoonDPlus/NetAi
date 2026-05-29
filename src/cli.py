from __future__ import annotations

import argparse
import json

from src.dataset_collect import collect_from_sitemaps, collect_from_urls, crawl_discovery_loop
from src.corpus_tools import load_texts_from_csv, save_stats_json, stats_from_csv
from src.generate import generate_sentences
from src.chatbot import MemoryChatbot
from src.instruct_import import import_instruct_repo
from src.api import run_api_server


def _load_numpy_stack():
    import numpy as np

    from src.features import BagOfWordsVectorizer
    from src.metrics import classification_report
    from src.mlp import BinaryCrossEntropy, MLPBinaryClassifier
    from src.text_data import load_labeled_text_csv, train_val_test_split_text

    return np, BagOfWordsVectorizer, classification_report, BinaryCrossEntropy, MLPBinaryClassifier, load_labeled_text_csv, train_val_test_split_text


def _to_xy(texts, labels, vec, np):
    x = vec.transform(texts)
    y = np.array(labels, dtype=float).reshape(-1, 1)
    return x, y


def _train_text(args: argparse.Namespace) -> None:
    (
        np,
        BagOfWordsVectorizer,
        classification_report,
        BinaryCrossEntropy,
        MLPBinaryClassifier,
        load_labeled_text_csv,
        train_val_test_split_text,
    ) = _load_numpy_stack()
    texts, labels = load_labeled_text_csv(args.data)
    x_train_t, y_train_l, x_val_t, y_val_l, x_test_t, y_test_l = train_val_test_split_text(
        texts, labels, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    vec = BagOfWordsVectorizer(max_features=args.max_features)
    vec.fit(x_train_t)

    x_train, y_train = _to_xy(x_train_t, y_train_l, vec, np)
    x_val, y_val = _to_xy(x_val_t, y_val_l, vec, np)
    x_test, y_test = _to_xy(x_test_t, y_test_l, vec, np)

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
    (
        np,
        BagOfWordsVectorizer,
        classification_report,
        _,
        MLPBinaryClassifier,
        load_labeled_text_csv,
        _,
    ) = _load_numpy_stack()
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
        count = collect_from_urls(
            urls,
            args.out_csv,
            min_chars=args.min_chars,
            delay_sec=args.delay_sec,
            timeout=args.timeout,
            retries=args.retries,
            verbose=args.verbose,
            ignore_robots=args.ignore_robots,
        )
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            sitemaps = [line.strip() for line in f if line.strip()]
        count = collect_from_sitemaps(
            sitemaps,
            args.out_csv,
            max_urls=args.max_urls,
            verbose=args.verbose,
            ignore_robots=args.ignore_robots,
        )
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
    corpus_texts = load_texts_from_csv(args.corpus_csv) if args.corpus_csv else None
    result = bot.respond(
        args.message,
        corpus_texts=corpus_texts,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _chat_learn(args: argparse.Namespace) -> None:
    bot = MemoryChatbot(memory_path=args.memory_path)
    bot.learn_pair(args.user_message, args.assistant_message)
    print(json.dumps({"saved": True, "memory_path": args.memory_path, "items": len(bot.pairs)}, ensure_ascii=False, indent=2))


def _import_instruct(args: argparse.Namespace) -> None:
    result = import_instruct_repo(args.repo_dir, args.memory_path)
    result["memory_path"] = args.memory_path
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _crawl_learn(args: argparse.Namespace) -> None:
    with open(args.input, "r", encoding="utf-8") as f:
        seeds = [line.strip() for line in f if line.strip()]
    result = crawl_discovery_loop(
        seeds,
        args.out_csv,
        min_chars=args.min_chars,
        delay_sec=args.delay_sec,
        timeout=args.timeout,
        retries=args.retries,
        verbose=args.verbose,
        ignore_robots=args.ignore_robots,
        ask_every=args.ask_every,
        workers=args.workers,
        db_path=args.db_path,
        save_every=args.save_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _serve_api(args: argparse.Namespace) -> None:
    run_api_server(
        host=args.host,
        port=args.port,
        memory_path=args.memory_path,
        corpus_csv=args.corpus_csv,
        db_path=args.db_path,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )

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
    collect_p.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for slow sites")
    collect_p.add_argument("--retries", type=int, default=2, help="Fetch retries before skipping a URL")
    collect_p.add_argument("--max-urls", type=int, default=200)
    collect_p.add_argument("--ignore-robots", action="store_true")
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
    chat_p.add_argument("--corpus-csv", default="", help="Optional crawled corpus CSV for automatic answers")
    chat_p.add_argument("--top-k", type=int, default=3, help="Number of corpus sentences to stitch into an answer")
    chat_p.add_argument("--min-similarity", type=float, default=0.15, help="Minimum similarity required before using memory/corpus answers")
    chat_p.set_defaults(func=_chat_reply)

    learn_p = sp.add_parser("learn-chat")
    learn_p.add_argument("--user-message", required=True)
    learn_p.add_argument("--assistant-message", required=True)
    learn_p.add_argument("--memory-path", default="data/chat_memory.json")
    learn_p.set_defaults(func=_chat_learn)

    instruct_p = sp.add_parser("import-instruct")
    instruct_p.add_argument("--repo-dir", required=True, help="Path to cloned instruct dataset repo")
    instruct_p.add_argument("--memory-path", default="data/chat_memory.json")
    instruct_p.set_defaults(func=_import_instruct)

    crawl_p = sp.add_parser("crawl-learn")
    crawl_p.add_argument("--input", required=True, help="Seed URL file (one URL per line)")
    crawl_p.add_argument("--out-csv", default="data/raw_web_text.csv")
    crawl_p.add_argument("--min-chars", type=int, default=200)
    crawl_p.add_argument("--delay-sec", type=float, default=1.0)
    crawl_p.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for slow sites")
    crawl_p.add_argument("--retries", type=int, default=2, help="Fetch retries before skipping a URL")
    crawl_p.add_argument("--ignore-robots", action="store_true")
    crawl_p.add_argument("--ask-every", type=int, default=100)
    crawl_p.add_argument("--workers", type=int, default=8, help="Concurrent requests for faster crawling")
    crawl_p.add_argument("--db-path", default="data/netai.db", help="SQLite DB for crawl checkpoints/pages")
    crawl_p.add_argument("--save-every", type=int, default=500, help="Save crawl snapshot to DB every N scanned URLs while still running")
    crawl_p.add_argument("--verbose", action="store_true")
    crawl_p.set_defaults(func=_crawl_learn)

    api_p = sp.add_parser("serve-api")
    api_p.add_argument("--host", default="127.0.0.1")
    api_p.add_argument("--port", type=int, default=8000)
    api_p.add_argument("--memory-path", default="data/chat_memory.json")
    api_p.add_argument("--corpus-csv", default="data/raw_web_text.csv")
    api_p.add_argument("--db-path", default="data/netai.db")
    api_p.add_argument("--top-k", type=int, default=3)
    api_p.add_argument("--min-similarity", type=float, default=0.15)
    api_p.set_defaults(func=_serve_api)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
