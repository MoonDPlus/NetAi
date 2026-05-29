from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.chatbot import MemoryChatbot
from src.corpus_tools import load_texts_from_csv, stats_from_csv
from src.storage import crawl_db_stats


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class NetAiAPIHandler(BaseHTTPRequestHandler):
    memory_path = "data/chat_memory.json"
    corpus_csv = "data/raw_web_text.csv"
    db_path = "data/netai.db"
    top_k = 3
    min_similarity = 0.15

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            _json_response(self, 200, {"ok": True, "service": "NetAi"})
            return
        if parsed.path == "/stats":
            payload: dict[str, object] = {"db": crawl_db_stats(self.db_path)}
            if Path(self.corpus_csv).exists():
                payload["corpus"] = stats_from_csv(self.corpus_csv)
            else:
                payload["corpus"] = None
            _json_response(self, 200, payload)
            return
        if parsed.path == "/chat":
            query = parse_qs(parsed.query)
            message = (query.get("message") or [""])[0]
            self._handle_chat({"message": message})
            return
        _json_response(self, 404, {"error": "not_found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            self._handle_chat(self._read_json())
            return
        _json_response(self, 404, {"error": "not_found"})

    def _handle_chat(self, payload: dict) -> None:
        message = str(payload.get("message") or "").strip()
        if not message:
            _json_response(self, 400, {"error": "message is required"})
            return
        memory_path = str(payload.get("memory_path") or self.memory_path)
        corpus_csv = str(payload.get("corpus_csv") or self.corpus_csv)
        top_k = int(payload.get("top_k") or self.top_k)
        min_similarity = float(payload.get("min_similarity") or self.min_similarity)

        corpus_texts = load_texts_from_csv(corpus_csv) if Path(corpus_csv).exists() else None
        bot = MemoryChatbot(memory_path=memory_path)
        result = bot.respond(message, corpus_texts=corpus_texts, top_k=top_k, min_similarity=min_similarity)
        _json_response(self, 200, result)


def create_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    memory_path: str = "data/chat_memory.json",
    corpus_csv: str = "data/raw_web_text.csv",
    db_path: str = "data/netai.db",
    top_k: int = 3,
    min_similarity: float = 0.15,
) -> ThreadingHTTPServer:
    handler_cls = type(
        "ConfiguredNetAiAPIHandler",
        (NetAiAPIHandler,),
        {
            "memory_path": memory_path,
            "corpus_csv": corpus_csv,
            "db_path": db_path,
            "top_k": top_k,
            "min_similarity": min_similarity,
        },
    )
    return ThreadingHTTPServer((host, port), handler_cls)


def run_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    memory_path: str = "data/chat_memory.json",
    corpus_csv: str = "data/raw_web_text.csv",
    db_path: str = "data/netai.db",
    top_k: int = 3,
    min_similarity: float = 0.15,
) -> None:
    server = create_api_server(
        host=host,
        port=port,
        memory_path=memory_path,
        corpus_csv=corpus_csv,
        db_path=db_path,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    print(json.dumps({"serving": True, "host": host, "port": port}, ensure_ascii=False))
    try:
        server.serve_forever()
    finally:
        server.server_close()
