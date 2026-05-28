import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
