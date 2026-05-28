import csv
import tempfile
import unittest

from src.corpus_tools import load_texts_from_csv, stats_from_csv


class TestCorpusTools(unittest.TestCase):
    def test_stats_from_csv(self):
        content = "url,text\nhttps://a,سلام دنیا. این یک تست است!\n"
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", encoding="utf-8", delete=True) as f:
            f.write(content)
            f.flush()
            stats = stats_from_csv(f.name)
        self.assertGreaterEqual(stats["corpus"]["total_words"], 4)
        self.assertGreaterEqual(stats["corpus"]["total_sentences"], 2)

    def test_load_texts_from_csv_raises_field_limit_for_large_crawl_rows(self):
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(64)
            large_text = "هوش مصنوعی " * 50
            with tempfile.NamedTemporaryFile("w+", suffix=".csv", encoding="utf-8", newline="", delete=True) as f:
                writer = csv.DictWriter(f, fieldnames=["url", "text"])
                writer.writeheader()
                writer.writerow({"url": "https://example.com", "text": large_text})
                f.flush()
                texts = load_texts_from_csv(f.name)
            self.assertEqual(texts, [large_text.strip()])
        finally:
            csv.field_size_limit(previous_limit)


if __name__ == "__main__":
    unittest.main()
