from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


class BagOfWordsVectorizer:
    def __init__(self, max_features: int = 2000) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in TOKEN_RE.findall(text)]

    def fit(self, texts: list[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        most_common = counter.most_common(self.max_features)
        self.vocab = {token: i for i, (token, _) in enumerate(most_common)}

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for i, text in enumerate(texts):
            for token in self._tokenize(text):
                j = self.vocab.get(token)
                if j is not None:
                    x[i, j] += 1.0
        return x

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str | Path) -> None:
        payload = {"max_features": self.max_features, "vocab": self.vocab}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BagOfWordsVectorizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(max_features=int(payload["max_features"]))
        obj.vocab = {str(k): int(v) for k, v in payload["vocab"].items()}
        return obj
