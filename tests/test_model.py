import tempfile
import unittest

import numpy as np

from src.mlp import MLPBinaryClassifier


class TestModel(unittest.TestCase):
    def test_forward_shape(self):
        model = MLPBinaryClassifier(seed=1)
        x = np.zeros((4, 2), dtype=float)
        y = model.forward(x)
        self.assertEqual(y.shape, (4, 1))

    def test_save_load(self):
        model1 = MLPBinaryClassifier(seed=1)
        model2 = MLPBinaryClassifier(seed=2)

        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        out1_before = model1.forward(x)
        out2_before = model2.forward(x)
        self.assertFalse(np.allclose(out1_before, out2_before))

        with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
            model1.save(tmp.name)
            model2.load(tmp.name)

        out2_after = model2.forward(x)
        self.assertTrue(np.allclose(out1_before, out2_after))


if __name__ == "__main__":
    unittest.main()
