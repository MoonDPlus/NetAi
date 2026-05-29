import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.dataset_collect import crawl_discovery_loop, encode_url, extract_clean_text, extract_links
from src.storage import crawl_db_stats


class TestDatasetCollect(unittest.TestCase):
    def test_extract_clean_text(self):
        html = "<html><head><style>.x{}</style></head><body><h1>سلام</h1><script>x=1</script><p>دنیا</p></body></html>"
        text = extract_clean_text(html)
        self.assertIn("سلام", text)
        self.assertIn("دنیا", text)
        self.assertNotIn("x=1", text)

    def test_extract_links_resolves_relative(self):
        html = '<a href="/a">A</a><a href="https://example.com/b?q=1">B</a><a href="mailto:x@y">M</a>'
        links = extract_links(html, "https://example.com/start")
        self.assertEqual(links, ["https://example.com/a", "https://example.com/b?q=1"])

    def test_encode_url_quotes_persian_query(self):
        url = encode_url("https://abadis.ir/dehkhoda/?ch=ا&pn=10")
        self.assertEqual(url, "https://abadis.ir/dehkhoda/?ch=%D8%A7&pn=10")
        url.encode("ascii")

    def test_encode_url_quotes_persian_path(self):
        url = encode_url("https://fa.wiktionary.org/wiki/گربه")
        self.assertEqual(url, "https://fa.wiktionary.org/wiki/%DA%AF%D8%B1%D8%A8%D9%87")
        url.encode("ascii")

    def test_extract_clean_text_falls_back_when_html_parser_asserts(self):
        html = '<html><body><script>bad()</script><p>سلام دنیا</p></body></html>'
        with patch("src.dataset_collect._TextExtractor.feed", side_effect=AssertionError("bad marked section")):
            text = extract_clean_text(html)
        self.assertIn("سلام دنیا", text)
        self.assertNotIn("bad()", text)

    def test_extract_links_falls_back_when_html_parser_asserts(self):
        html = '<a href="/fallback">fallback</a>'
        with patch("src.dataset_collect._LinkExtractor.feed", side_effect=AssertionError("bad marked section")):
            links = extract_links(html, "https://example.com/base")
        self.assertEqual(links, ["https://example.com/fallback"])

    def test_crawl_discovery_loop_discovers_links_with_workers(self):
        pages = {
            "https://example.com/start": '<p>این متن شروع برای ذخیره شدن کافی است</p><a href="/next">next</a>',
            "https://example.com/next": "<p>این متن صفحه بعدی برای ذخیره شدن کافی است</p>",
        }

        def fake_fetch(url, user_agent, timeout):
            return pages[url]

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "crawl.csv"
            db = Path(d) / "crawl.db"
            with patch("src.dataset_collect._fetch", side_effect=fake_fetch):
                stats = crawl_discovery_loop(
                    ["https://example.com/start"],
                    out,
                    min_chars=10,
                    workers=2,
                    ask_every=0,
                    ignore_robots=True,
                    db_path=db,
                    save_every=1,
                )
            self.assertEqual(stats["scanned"], 2)
            with out.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            db_stats = crawl_db_stats(db)
            self.assertEqual(db_stats["pages"], 2)
            self.assertEqual(db_stats["last_scanned"], 2)


if __name__ == "__main__":
    unittest.main()
