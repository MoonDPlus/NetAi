import tempfile
import unittest
import json
from pathlib import Path

from src.chatbot import MemoryChatbot, answer_from_corpus


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

    def test_respond_from_corpus_without_manual_training(self):
        with tempfile.TemporaryDirectory() as d:
            bot = MemoryChatbot(memory_path=Path(d) / "missing_mem.json")
            texts = ["هوش مصنوعی شاخه‌ای از علوم کامپیوتر است. یادگیری ماشین به مدل‌ها کمک می‌کند از داده یاد بگیرند."]
            res = bot.respond("هوش مصنوعی چیست", corpus_texts=texts)
            self.assertEqual(res["mode"], "corpus")
            self.assertIn("هوش مصنوعی", res["reply"])

    def test_answer_from_corpus_ranks_relevant_sentence(self):
        texts = ["گربه حیوان خانگی است. شبکه عصبی برای یادگیری عمیق استفاده می‌شود."]
        res = answer_from_corpus("یادگیری عمیق", texts, top_k=1)
        self.assertIn("یادگیری عمیق", res["reply"])


if __name__ == "__main__":
    unittest.main()
