from __future__ import annotations

import random
import re
from collections import defaultdict

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"[.!؟!?\n]+")


def build_bigram_chain(texts: list[str]) -> tuple[dict[str, list[str]], list[str]]:
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
    chain, starts = build_bigram_chain(texts)
    if not starts:
        return []

    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n_sentences):
        current = rng.choice(starts)
        words = [current]
        for _ in range(max_words - 1):
            next_options = chain.get(current)
            if not next_options:
                break
            current = rng.choice(next_options)
            words.append(current)
        out.append(" ".join(words))
    return out
