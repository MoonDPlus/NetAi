from __future__ import annotations

import json
import re
from pathlib import Path

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


class MemoryChatbot:
    def __init__(self, memory_path: str | Path = "data/chat_memory.json") -> None:
        self.memory_path = Path(memory_path)
        self.pairs: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if self.memory_path.exists():
            self.pairs = json.loads(self.memory_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.write_text(json.dumps(self.pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    def learn_pair(self, user_text: str, assistant_text: str) -> None:
        self.pairs.append({"user": user_text.strip(), "assistant": assistant_text.strip()})
        self._save()

    def respond(self, user_text: str) -> dict[str, object]:
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

        if best is None or best_score <= 0:
            reply = (
                "هنوز نمونه مشابهی در حافظه ندارم. "
                "لطفاً پاسخ درست یا ترجیحی‌ات رو با دستور learn-chat ذخیره کن تا ازش یاد بگیرم."
            )
            return {"reply": reply, "similarity": 0.0, "source": None}

        return {
            "reply": best["assistant"],
            "similarity": round(best_score, 4),
            "source": best["user"],
        }
