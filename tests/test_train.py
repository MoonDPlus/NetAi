import tempfile
import unittest

from src.train import train


class TestTrain(unittest.TestCase):
    def test_train_writes_model(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            train(
                epochs=5,
                lr=0.05,
                seed=1,
                batch_size=8,
                repeats=20,
                noise=0.01,
                save_path=tmp.name,
                optimizer="adam",
                patience=3,
            )
            with open(tmp.name, "r", encoding="utf-8") as f:
                content = f.read().strip()
            self.assertTrue(content.startswith("{"))


if __name__ == "__main__":
    unittest.main()
