from __future__ import annotations

import csv
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

DEFAULT_UA = "NetAiDatasetBot/0.1 (+https://example.local)"
FALLBACK_ALLOWED_UA = "*"


def encode_url(url: str) -> str:
    """Convert a possibly-unicode IRI to an ASCII-safe URI for urllib."""
    parts = urlsplit(url.strip())
    netloc = parts.netloc
    if parts.hostname:
        host = parts.hostname.encode("idna").decode("ascii")
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        userinfo = ""
        if parts.username:
            userinfo = quote(parts.username, safe="")
            if parts.password:
                userinfo += f":{quote(parts.password, safe='')}"
            userinfo += "@"
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{userinfo}{host}{port}"

    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&;%+,:/?")
    fragment = quote(parts.fragment, safe="=&;%+,:/?")
    return urlunsplit((parts.scheme, netloc, path, query, fragment))


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


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value.strip())


def _allowed(
    url: str,
    user_agent: str,
    *,
    strict: bool = False,
    verbose: bool = False,
    ignore_robots: bool = False,
) -> bool:
    if ignore_robots:
        return True

    safe_url = encode_url(url)
    parsed = urlparse(safe_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        # Try with exact UA first, then fallback to wildcard rules.
        allowed = rp.can_fetch(user_agent, safe_url)
        if not allowed:
            allowed = rp.can_fetch(FALLBACK_ALLOWED_UA, safe_url)
        return allowed
    except Exception as exc:
        if verbose:
            print(f"[robots unavailable] {robots_url} -> {exc}")
        # Fail-open by default so connectivity/robots fetch issues do not block all crawling.
        return not strict


def _fetch(url: str, user_agent: str, timeout: float) -> str:
    req = Request(encode_url(url), headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def extract_clean_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    text = unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


def extract_links(html: str, base_url: str) -> list[str]:
    parser = _LinkExtractor()
    parser.feed(html)
    out: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            continue
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out



def _write_crawl_rows(out_path: Path, rows: list[CrawlItem]) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "text"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"url": row.url, "text": row.text})


def _crawl_page(
    url: str,
    *,
    user_agent: str,
    min_chars: int,
    timeout: float,
    verbose: bool,
    ignore_robots: bool,
) -> tuple[str, CrawlItem | None, list[str]]:
    if not _allowed(url, user_agent=user_agent, verbose=verbose, ignore_robots=ignore_robots):
        if verbose:
            print(f"[skip robots] {url}")
        return url, None, []

    try:
        html = _fetch(url, user_agent=user_agent, timeout=timeout)
    except Exception as exc:
        if verbose:
            print(f"[skip fetch] {url} -> {exc}")
        return url, None, []

    text = extract_clean_text(html)
    item = None
    if len(text) >= min_chars:
        item = CrawlItem(url=url, text=text)
        if verbose:
            print(f"[ok] {url} len={len(text)}")
    elif verbose:
        print(f"[skip short] {url} len={len(text)}")

    return url, item, extract_links(html, url)

def crawl_discovery_loop(
    seed_urls: list[str],
    out_csv: str | Path,
    *,
    user_agent: str = DEFAULT_UA,
    min_chars: int = 200,
    delay_sec: float = 1.0,
    timeout: float = 10.0,
    verbose: bool = False,
    ignore_robots: bool = False,
    ask_every: int = 100,
    workers: int = 8,
) -> dict[str, int | bool]:
    from concurrent.futures import ThreadPoolExecutor

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queue: list[str] = [u.strip() for u in seed_urls if u.strip()]
    queued: set[str] = set(queue)
    seen: set[str] = set()
    rows: list[CrawlItem] = []
    scanned = 0
    stopped_by_user = False
    max_workers = max(1, workers)

    def submit_next(executor: ThreadPoolExecutor, in_flight: dict[Future, str]) -> None:
        while queue and len(in_flight) < max_workers:
            url = queue.pop(0)
            queued.discard(url)
            if url in seen:
                continue
            seen.add(url)
            future = executor.submit(
                _crawl_page,
                url,
                user_agent=user_agent,
                min_chars=min_chars,
                timeout=timeout,
                verbose=verbose,
                ignore_robots=ignore_robots,
            )
            in_flight[future] = url
            if delay_sec > 0:
                time.sleep(delay_sec)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        in_flight: dict[Future, str] = {}
        submit_next(executor, in_flight)

        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                in_flight.pop(future, None)
                scanned += 1
                _, item, links = future.result()
                if item is not None:
                    rows.append(item)
                for new_url in links:
                    if new_url not in seen and new_url not in queued:
                        queue.append(new_url)
                        queued.add(new_url)

                if ask_every > 0 and scanned % ask_every == 0:
                    _write_crawl_rows(out_path, rows)
                    prompt = f"Scanned {scanned} links (saved={len(rows)}, queued={len(queue)}). Continue? [y/N]: "
                    answer = input(prompt).strip().lower()
                    if answer not in {"y", "yes"}:
                        stopped_by_user = True
                        queue.clear()
                        for pending in in_flight:
                            pending.cancel()
                        in_flight.clear()
                        break

            if stopped_by_user:
                break
            submit_next(executor, in_flight)

    _write_crawl_rows(out_path, rows)
    return {"scanned": scanned, "saved": len(rows), "queued": len(queue), "stopped_by_user": stopped_by_user}


def collect_from_urls(
    urls: list[str],
    out_csv: str | Path,
    *,
    user_agent: str = DEFAULT_UA,
    min_chars: int = 200,
    delay_sec: float = 1.0,
    timeout: float = 10.0,
    verbose: bool = False,
    ignore_robots: bool = False,
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

        if not _allowed(url, user_agent=user_agent, verbose=verbose, ignore_robots=ignore_robots):
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

    _write_crawl_rows(out_path, rows)

    return len(rows)


def collect_from_sitemaps(
    sitemap_urls: list[str],
    out_csv: str | Path,
    *,
    max_urls: int = 200,
    user_agent: str = DEFAULT_UA,
    verbose: bool = False,
    ignore_robots: bool = False,
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

    return collect_from_urls(
        urls,
        out_csv,
        user_agent=user_agent,
        verbose=verbose,
        ignore_robots=ignore_robots,
    )
