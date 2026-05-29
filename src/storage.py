from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from src.dataset_collect import CrawlItem


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crawl_pages (
            url TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            text_len INTEGER NOT NULL,
            saved_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS crawl_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned INTEGER NOT NULL,
            saved INTEGER NOT NULL,
            queued INTEGER NOT NULL,
            in_flight INTEGER NOT NULL,
            stopped_by_user INTEGER NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.commit()


def save_crawl_snapshot(
    db_path: str | Path,
    rows: list[CrawlItem],
    *,
    scanned: int,
    queued: int,
    in_flight: int,
    stopped_by_user: bool = False,
) -> None:
    now = time.time()
    with connect_db(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO crawl_pages(url, text, text_len, saved_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                text=excluded.text,
                text_len=excluded.text_len,
                saved_at=excluded.saved_at
            """,
            [(row.url, row.text, len(row.text), now) for row in rows],
        )
        conn.execute(
            """
            INSERT INTO crawl_checkpoints(scanned, saved, queued, in_flight, stopped_by_user, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (scanned, len(rows), queued, in_flight, int(stopped_by_user), now),
        )
        conn.commit()


def crawl_db_stats(db_path: str | Path) -> dict[str, int | float | None]:
    path = Path(db_path)
    if not path.exists():
        return {"pages": 0, "total_chars": 0, "last_scanned": None, "last_saved": None, "last_checkpoint_at": None}
    with connect_db(path) as conn:
        pages, total_chars = conn.execute("SELECT COUNT(*), COALESCE(SUM(text_len), 0) FROM crawl_pages").fetchone()
        checkpoint = conn.execute(
            """
            SELECT scanned, saved, created_at
            FROM crawl_checkpoints
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if checkpoint is None:
        return {"pages": pages, "total_chars": total_chars, "last_scanned": None, "last_saved": None, "last_checkpoint_at": None}
    return {
        "pages": pages,
        "total_chars": total_chars,
        "last_scanned": checkpoint[0],
        "last_saved": checkpoint[1],
        "last_checkpoint_at": checkpoint[2],
    }
