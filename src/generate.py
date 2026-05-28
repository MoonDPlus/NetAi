from __future__ import annotations

import random
import re
from collections import defaultdict

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!؟!؟?\n]+")
END = "<END>"


def build_trigram_chain(texts: list[str]) -> tuple[dict[tuple[str, str], list[str]], list[tuple[str, str]]]:
    chain: dict[tuple[str, str], list[str]] = defaultdict(list)
    starts: list[tuple[str, str]] = []
    for text in texts:
        for sent in [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]:
            toks = [t.lower() for t in TOKEN_RE.findall(sent)]
            if len(toks) < 3:
                continue
            starts.append((toks[0], toks[1]))
            extended = toks + [END]
            for a, b, c in zip(extended, extended[1:], extended[2:], strict=False):
                chain[(a, b)].append(c)
    return chain, starts


def build_bigram_chain(texts: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    # Kept for backward compatibility with earlier tests/users.
    chain: dict[str, list[str]] = defaultdict(list)
    starts: list[str] = []
    for text in texts:
        for sent in [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]:
            toks = [t.lower() for t in TOKEN_RE.findall(sent)]
            if len(toks) < 2:
                continue
            starts.append(toks[0])
            for a, b in zip(toks, toks[1:], strict=False):
                chain[a].append(b)
    return chain, starts


def generate_sentences(texts: list[str], n_sentences: int = 5, max_words: int = 12, seed: int = 42) -> list[str]:
    chain, starts = build_trigram_chain(texts)
    if not starts:
        return []

    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n_sentences):
        first, second = rng.choice(starts)
        words = [first, second]
        while len(words) < max_words:
            next_options = chain.get((words[-2], words[-1]))
            if not next_options:
                break
            nxt = rng.choice(next_options)
            if nxt == END:
                break
            words.append(nxt)
        out.append(" ".join(words))
    return out
