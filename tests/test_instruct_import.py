from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.instruct_import import import_instruct_repo


class TestInstructImport(unittest.TestCase):
    def test_import_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sample = root / "sample.jsonl"
            sample.write_text(
                '{"instruction":"سلام","input":"حالت چطوره؟","output":"خوبم"}\n'
                '{"instruction":"A","output":"B"}\n',
                encoding="utf-8",
            )
            out = root / "memory.json"
            stats = import_instruct_repo(root, out)
            self.assertEqual(stats["pairs"], 2)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(len(data["pairs"]), 2)


if __name__ == "__main__":
    unittest.main()
