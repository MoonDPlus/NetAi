from __future__ import annotations

import csv
import json
from pathlib import Path


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_triplet(obj: dict) -> tuple[str, str, str]:
    instruction = _as_text(obj.get("instruction") or obj.get("prompt") or obj.get("question"))
    input_text = _as_text(obj.get("input") or obj.get("context"))
    output = _as_text(obj.get("output") or obj.get("answer") or obj.get("response"))
    return instruction, input_text, output


def _iter_objects(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            for key in ("data", "items", "examples"):
                section = data.get(key)
                if isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict):
                            yield item
        return

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)


def import_instruct_repo(repo_dir: str | Path, out_memory_json: str | Path) -> dict[str, int]:
    root = Path(repo_dir)
    if not root.exists():
        raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

    pairs: list[dict[str, str]] = []
    files_used = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        added_before = len(pairs)
        try:
            for obj in _iter_objects(path):
                if not isinstance(obj, dict):
                    continue
                instruction, input_text, output = _extract_triplet(obj)
                if not instruction or not output:
                    continue
                user_message = f"{instruction}\n{input_text}".strip() if input_text else instruction
                pairs.append({"user": user_message, "assistant": output})
        except Exception:
            continue
        if len(pairs) > added_before:
            files_used += 1

    out_path = Path(out_memory_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"pairs": pairs}, f, ensure_ascii=False, indent=2)

    return {"pairs": len(pairs), "files_used": files_used}
