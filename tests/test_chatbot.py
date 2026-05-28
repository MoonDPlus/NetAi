import tempfile
import unittest
import json
from pathlib import Path

from src.chatbot import MemoryChatbot


class TestMemoryChatbot(unittest.TestCase):
    def test_learn_and_respond(self):
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/mem.json"
            bot = MemoryChatbot(memory_path=path)
            bot.learn_pair("سلام حالت چطوره", "سلام، خوبم")
            res = bot.respond("سلام خوبی")
            self.assertIn("reply", res)
            self.assertGreaterEqual(res["similarity"], 0)

    def test_load_dict_pairs_format(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "mem.json"
            path.write_text(
                json.dumps({"pairs": [{"user": "سلام", "assistant": "سلام!"}]}, ensure_ascii=False),
                encoding="utf-8",
            )
            bot = MemoryChatbot(memory_path=path)
            res = bot.respond("سلام")
            self.assertEqual(res["reply"], "سلام!")


if __name__ == "__main__":
    unittest.main()
