import unittest

from src.text_data import train_val_test_split_text


class TestTextSplits(unittest.TestCase):
    def test_split_sizes(self):
        texts = [f"t{i}" for i in range(10)]
        labels = [i % 2 for i in range(10)]
        xtr, ytr, xv, yv, xte, yte = train_val_test_split_text(texts, labels, 0.2, 0.2, seed=1)
        self.assertEqual(len(xtr) + len(xv) + len(xte), 10)
        self.assertEqual(len(ytr) + len(yv) + len(yte), 10)
        self.assertGreaterEqual(len(xtr), 1)


if __name__ == "__main__":
    unittest.main()
