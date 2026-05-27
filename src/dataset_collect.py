from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

DEFAULT_UA = "NetAiDatasetBot/0.1 (+https://example.local)"


@dataclass
class CrawlItem:
    url: str
    text: str


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip += 1

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript"} and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data: str):
        if self._skip == 0:
            d = data.strip()
            if d:
                self.parts.append(d)


def _allowed(url: str, user_agent: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return False


def _fetch(url: str, user_agent: str, timeout: float) -> str:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def extract_clean_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


def collect_from_urls(
    urls: list[str],
    out_csv: str | Path,
    *,
    user_agent: str = DEFAULT_UA,
    min_chars: int = 200,
    delay_sec: float = 1.0,
    timeout: float = 10.0,
    verbose: bool = False,
) -> int:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[CrawlItem] = []
    seen: set[str] = set()

    for raw in urls:
        url = raw.strip()
        if not url or url in seen:
            continue
        seen.add(url)

        if not _allowed(url, user_agent=user_agent):
            if verbose:
                print(f"[skip robots] {url}")
            continue

        try:
            html = _fetch(url, user_agent=user_agent, timeout=timeout)
        except Exception as exc:
            if verbose:
                print(f"[skip fetch] {url} -> {exc}")
            continue

        text = extract_clean_text(html)
        if len(text) < min_chars:
            if verbose:
                print(f"[skip short] {url} len={len(text)}")
            continue

        rows.append(CrawlItem(url=url, text=text))
        if verbose:
            print(f"[ok] {url} len={len(text)}")
        time.sleep(delay_sec)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"url": row.url, "text": row.text})

    return len(rows)


def collect_from_sitemaps(
    sitemap_urls: list[str],
    out_csv: str | Path,
    *,
    max_urls: int = 200,
    user_agent: str = DEFAULT_UA,
    verbose: bool = False,
) -> int:
    urls: list[str] = []
    loc_re = re.compile(r"<loc>(.*?)</loc>", re.IGNORECASE)

    for sitemap_url in sitemap_urls:
        try:
            xml = _fetch(sitemap_url, user_agent=user_agent, timeout=15.0)
        except Exception as exc:
            if verbose:
                print(f"[skip sitemap] {sitemap_url} -> {exc}")
            continue

        for match in loc_re.findall(xml):
            u = match.strip()
            if u:
                urls.append(u)
                if len(urls) >= max_urls:
                    break
        if len(urls) >= max_urls:
            break

    return collect_from_urls(urls, out_csv, user_agent=user_agent, verbose=verbose)
