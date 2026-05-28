import tempfile
import unittest

from src.corpus_tools import stats_from_csv


class TestCorpusTools(unittest.TestCase):
    def test_stats_from_csv(self):
        content = "url,text\nhttps://a,سلام دنیا. این یک تست است!\n"
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", encoding="utf-8", delete=True) as f:
            f.write(content)
            f.flush()
            stats = stats_from_csv(f.name)
        self.assertGreaterEqual(stats["corpus"]["total_words"], 4)
        self.assertGreaterEqual(stats["corpus"]["total_sentences"], 2)
