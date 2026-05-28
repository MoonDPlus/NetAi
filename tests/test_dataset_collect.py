import unittest

from src.dataset_collect import encode_url, extract_clean_text, extract_links


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


if __name__ == "__main__":
    unittest.main()
