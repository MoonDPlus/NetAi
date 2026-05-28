from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!؟!?\n]+")


def ensure_large_csv_fields() -> None:
    """Allow reading crawled pages whose text field is larger than csv's small default."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_texts_from_csv(path: str | Path, text_column: str = "text") -> list[str]:
    ensure_large_csv_fields()
    texts: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get(text_column) or "").strip()
            if t:
                texts.append(t)
    return texts


def corpus_stats(texts: list[str]) -> dict[str, int | float]:
    all_tokens: list[str] = []
    sentence_count = 0
    for text in texts:
        tokens = [t.lower() for t in TOKEN_RE.findall(text)]
        all_tokens.extend(tokens)
        sentence_count += len([s for s in SENT_SPLIT_RE.split(text) if s.strip()])

    unique_words = len(set(all_tokens))
    total_words = len(all_tokens)
    avg_sentence_len = (total_words / sentence_count) if sentence_count else 0.0
    return {
        "documents": len(texts),
        "total_words": total_words,
        "unique_words": unique_words,
        "total_sentences": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 2),
    }


def generation_capacity_estimate(texts: list[str]) -> dict[str, int]:
    transitions: dict[str, set[str]] = defaultdict(set)
    starts: set[str] = set()

    for text in texts:
        for sent in [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]:
            toks = [t.lower() for t in TOKEN_RE.findall(sent)]
            if not toks:
                continue
            starts.add(toks[0])
            for a, b in zip(toks, toks[1:], strict=False):
                transitions[a].add(b)

    edge_count = sum(len(v) for v in transitions.values())
    return {
        "sentence_start_options": len(starts),
        "word_transition_options": edge_count,
    }


def stats_from_csv(path: str | Path) -> dict[str, object]:
    texts = load_texts_from_csv(path)
    return {
        "corpus": corpus_stats(texts),
        "generation_estimate": generation_capacity_estimate(texts),
    }


def save_stats_json(stats: dict[str, object], out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
