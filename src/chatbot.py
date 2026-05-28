from __future__ import annotations

import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!؟?])\s+|[\n\r]+")


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for part in SENT_SPLIT_RE.split(text):
        sent = re.sub(r"\s+", " ", part).strip(" \t\n\r.,،؛:؛")
        if len(sent) >= 20:
            sentences.append(sent)
    return sentences


def answer_from_corpus(user_text: str, texts: list[str], *, top_k: int = 3) -> dict[str, object]:
    query_tokens = _tokenize(user_text)
    if not query_tokens:
        return {"reply": "پیام قابل تحلیل نبود.", "similarity": 0.0, "source": None, "mode": "corpus"}

    scored: list[tuple[float, str]] = []
    for text in texts:
        for sentence in split_sentences(text):
            toks = _tokenize(sentence)
            if not toks:
                continue
            overlap = len(query_tokens & toks)
            if overlap == 0:
                continue
            score = overlap / (len(query_tokens) ** 0.5 * len(toks) ** 0.5)
            scored.append((score, sentence))

    if not scored:
        return {
            "reply": "در کورپوس فعلی جمله مرتبطی پیدا نکردم. اول با crawl-learn داده بیشتری جمع کن.",
            "similarity": 0.0,
            "source": None,
            "mode": "corpus",
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for score, sentence in scored:
        if sentence in seen:
            continue
        selected.append(sentence)
        seen.add(sentence)
        if len(selected) >= top_k:
            break

    reply = " ".join(selected)
    return {"reply": reply, "similarity": round(scored[0][0], 4), "source": selected[0], "mode": "corpus"}


class MemoryChatbot:
    def __init__(self, memory_path: str | Path = "data/chat_memory.json") -> None:
        self.memory_path = Path(memory_path)
        self.pairs: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if self.memory_path.exists():
            raw = json.loads(self.memory_path.read_text(encoding="utf-8"))
            # Backward/forward compatibility:
            # - old format: [ {"user": "...", "assistant": "..."} ]
            # - new format: { "pairs": [ ... ] }
            if isinstance(raw, dict):
                raw = raw.get("pairs", [])
            if not isinstance(raw, list):
                raw = []

            cleaned: list[dict[str, str]] = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                user = item.get("user")
                assistant = item.get("assistant")
                if isinstance(user, str) and isinstance(assistant, str):
                    cleaned.append({"user": user, "assistant": assistant})
            self.pairs = cleaned

    def _save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    def learn_pair(self, user_text: str, assistant_text: str) -> None:
        self.pairs.append({"user": user_text.strip(), "assistant": assistant_text.strip()})
        self._save()

    def respond(self, user_text: str, corpus_texts: list[str] | None = None, *, top_k: int = 3) -> dict[str, object]:
        query_tokens = _tokenize(user_text)
        best = None
        best_score = -1.0
        for item in self.pairs:
            toks = _tokenize(item["user"])
            if not toks:
                continue
            inter = len(query_tokens & toks)
            union = len(query_tokens | toks)
            score = inter / union if union else 0.0
            if score > best_score:
                best_score = score
                best = item

        if best is not None and best_score > 0:
            return {
                "reply": best["assistant"],
                "similarity": round(best_score, 4),
                "source": best["user"],
                "mode": "memory",
            }

        if corpus_texts:
            return answer_from_corpus(user_text, corpus_texts, top_k=top_k)

        reply = (
            "هنوز نمونه مشابهی در حافظه یا کورپوس ندارم. "
            "با crawl-learn داده جمع کن یا با --corpus-csv کورپوس جمع‌شده را به chat بده."
        )
        return {"reply": reply, "similarity": 0.0, "source": None, "mode": "empty"}
