import tempfile
import unittest

from src.features import BagOfWordsVectorizer


class TestFeatures(unittest.TestCase):
    def test_fit_transform_shape(self):
        texts = ["a b", "b c"]
        vec = BagOfWordsVectorizer(max_features=10)
        x = vec.fit_transform(texts)
        self.assertEqual(x.shape[0], 2)
        self.assertGreaterEqual(x.shape[1], 2)

    def test_save_load(self):
        vec = BagOfWordsVectorizer(max_features=10)
        vec.fit(["hello world", "hello"])
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            vec.save(tmp.name)
            vec2 = BagOfWordsVectorizer.load(tmp.name)
        self.assertEqual(vec.vocab, vec2.vocab)


if __name__ == "__main__":
    unittest.main()
