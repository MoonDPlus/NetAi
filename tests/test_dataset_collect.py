import unittest

from src.dataset_collect import extract_clean_text


class TestDatasetCollect(unittest.TestCase):
    def test_extract_clean_text(self):
        html = "<html><head><style>.x{}</style></head><body><h1>سلام</h1><script>x=1</script><p>دنیا</p></body></html>"
        text = extract_clean_text(html)
        self.assertIn("سلام", text)
        self.assertIn("دنیا", text)
        self.assertNotIn("x=1", text)


if __name__ == "__main__":
    unittest.main()
