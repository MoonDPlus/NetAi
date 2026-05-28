import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from src.api import create_api_server


class TestAPI(unittest.TestCase):
    def test_chat_endpoint_answers(self):
        with tempfile.TemporaryDirectory() as d:
            corpus = Path(d) / "corpus.csv"
            with corpus.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["url", "text"])
                writer.writeheader()
                writer.writerow({"url": "local", "text": "Artificial intelligence learns from data. Persian text can be learned too."})

            server = create_api_server(host="127.0.0.1", port=0, corpus_csv=str(corpus), memory_path=str(Path(d) / "mem.json"))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                body = json.dumps({"message": "What is artificial intelligence?", "min_similarity": 0.05}).encode("utf-8")
                req = Request(
                    f"http://127.0.0.1:{port}/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(payload["mode"], "corpus")
                self.assertEqual(payload["language"], "en")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
