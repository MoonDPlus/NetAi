import tempfile
import unittest
from pathlib import Path

from src.dataset_collect import CrawlItem
from src.storage import crawl_db_stats, save_crawl_snapshot


class TestStorage(unittest.TestCase):
    def test_save_crawl_snapshot_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "netai.db"
            rows = [CrawlItem(url="https://example.com", text="hello world")]
            save_crawl_snapshot(db, rows, scanned=500, queued=10, in_flight=2)
            stats = crawl_db_stats(db)
            self.assertEqual(stats["pages"], 1)
            self.assertEqual(stats["last_scanned"], 500)
            self.assertEqual(stats["last_saved"], 1)


if __name__ == "__main__":
    unittest.main()
