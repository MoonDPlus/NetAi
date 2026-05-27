import unittest

from src.generate import generate_sentences


class TestGenerate(unittest.TestCase):
    def test_generate_non_empty(self):
        texts = ["سلام دنیا. این یک تست است", "هوش مصنوعی یاد میگیرد"]
        out = generate_sentences(texts, n_sentences=3, max_words=6, seed=1)
        self.assertEqual(len(out), 3)
        self.assertTrue(all(isinstance(x, str) and x for x in out))


if __name__ == "__main__":
    unittest.main()
