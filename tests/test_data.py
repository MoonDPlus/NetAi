import unittest

from src.data import make_xor_dataset, train_val_split


class TestData(unittest.TestCase):
    def test_xor_shape(self):
        x, y = make_xor_dataset(repeats=10)
        self.assertEqual(x.shape, (40, 2))
        self.assertEqual(y.shape, (40, 1))

    def test_train_val_split(self):
        x, y = make_xor_dataset(repeats=5)
        xtr, ytr, xv, yv = train_val_split(x, y, val_ratio=0.2, seed=1)
        self.assertEqual(len(xtr) + len(xv), len(x))
        self.assertEqual(len(ytr) + len(yv), len(y))


if __name__ == "__main__":
    unittest.main()
