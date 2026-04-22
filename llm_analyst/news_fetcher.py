"""Pull crypto/macro news headlines from free RSS feeds into the news store.

Why stdlib XML instead of ``feedparser``?
-----------------------------------------
``feedparser`` is the obvious choice but it's an extra dependency we don't
need for "read 50 RSS items a day". ``xml.etree.ElementTree`` with a small
namespace-aware helper handles both RSS 2.0 (``<item>``) and Atom
(``<entry>``) cleanly. If a feed changes format and our parser chokes, we
log the error to ``ingest_log`` and keep going — one bad source doesn't
kill the run.

Feed list
---------
Curated for **ETH 30-day direction** relevance, not breadth:
  - CoinTelegraph  — volume, reliable RSS, catches narrative shifts fast
  - Decrypt        — editorial, good on regulatory / ETF threads
  - The Block      — institutional focus (SEC, ETF flows, staking)
  - CoinDesk       — macro/policy overlap, slower but higher-signal

We deliberately skip Twitter/X (paid API + noise) and Reddit (moderation
ban surface area). Adding sources later = one line in ``FEEDS``.

Running
-------
    python -m llm_analyst.news_fetcher              # pull all feeds once
    python -m llm_analyst.news_fetcher --dry-run    # parse but don't write
"""
from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import requests

from llm_analyst.news_store import (
    DEFAULT_DB_PATH,
    connect,
    init_db,
    insert_headlines,
    record_ingest,
    stats,
)

USER_AGENT = (
    "eth-forecast-news-fetcher/0.1 "
    "(+https://etherforecast.live; research pipeline, no redistribution)"
)
TIMEOUT_SECONDS = 20

# Atom uses this namespace; RSS 2.0 doesn't, but including the map is
# harmless and means ``{atom}link`` below works for both.
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


@dataclass(frozen=True)
class Feed:
    source: str
    url: str
    # Some feeds are absolute-URL only in <guid>, others only in <link>.
    # ``url_priority`` controls which element we trust first.
    url_priority: tuple[str, ...] = ("link", "guid", "atom:link")


FEEDS: tuple[Feed, ...] = (
    Feed("cointelegraph", "https://cointelegraph.com/rss"),
    Feed("decrypt", "https://decrypt.co/feed"),
    Feed("theblock", "https://www.theblock.co/rss.xml"),
    Feed("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
)


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------


def _parse_datetime(raw: str | None) -> str | None:
    """Return an ISO-8601 UTC string, or None if the input is unparseable.

    RSS 2.0 uses RFC 822 (``Sun, 01 Jan 2024 12:00:00 +0000``); Atom uses
    RFC 3339 (``2024-01-01T12:00:00Z``). ``parsedate_to_datetime`` handles
    RFC 822 natively; Atom falls through to ``datetime.fromisoformat``.
    """
    if not raw:
        return None
    raw = raw.strip()
    # Atom / ISO-8601 path.
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except ValueError:
        pass
    # RFC 822 path.
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _text_or_none(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    text = element.text
    return text.strip() if text else None


def _find_first(root: ET.Element, paths: Iterable[str]) -> ET.Element | None:
    for path in paths:
        found = root.find(path, NAMESPACES)
        if found is not None:
            return found
    return None


def _extract_url(item: ET.Element, feed: Feed) -> str | None:
    """Return the best URL for this item, honouring ``feed.url_priority``."""
    for key in feed.url_priority:
        if key == "link":
            link_el = item.find("link")
            if link_el is not None and link_el.text and link_el.text.strip():
                return link_el.text.strip()
        elif key == "guid":
            guid_el = item.find("guid")
            if guid_el is not None and guid_el.text and guid_el.text.strip():
                candidate = guid_el.text.strip()
                if candidate.startswith(("http://", "https://")):
                    return candidate
        elif key == "atom:link":
            link_el = item.find("atom:link", NAMESPACES)
            if link_el is not None:
                href = link_el.get("href")
                if href:
                    return href.strip()
    return None


def parse_feed(xml_bytes: bytes, feed: Feed) -> list[dict]:
    """Parse an RSS 2.0 or Atom feed body into headline dicts.

    Returns the list; caller decides whether to insert. Invalid items are
    silently dropped — we never halt the whole feed on one bad entry.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"XML parse failed for {feed.url}: {exc}") from exc

    items: list[dict] = []
    channel_el = root.find("channel")
    channel = channel_el if channel_el is not None else root  # Atom: entries live on root.

    # RSS 2.0 <item>, Atom <entry>. Support both in one pass.
    for tag in ("item", "{http://www.w3.org/2005/Atom}entry"):
        for item in channel.iter(tag):
            title_el = _find_first(item, ("title", "atom:title"))
            title = _text_or_none(title_el)
            url = _extract_url(item, feed)
            if not title or not url:
                continue

            # RFC-822 or ISO-8601 depending on format.
            published_raw = _text_or_none(
                _find_first(item, ("pubDate", "atom:published", "atom:updated", "dc:date"))
            )
            published_iso = _parse_datetime(published_raw)
            if not published_iso:
                # Skip items without a parseable timestamp — we can't
                # date-gate them in the backtest harness.
                continue

            summary_raw = _text_or_none(
                _find_first(item, ("description", "atom:summary", "content:encoded"))
            )
            # Strip trivial HTML tags so the summary doesn't bloat the LLM
            # prompt with <p>, <a href=...> noise. We don't need perfect
            # HTML-to-text — the goal is readable bullets.
            summary = _strip_html(summary_raw) if summary_raw else None

            categories_el = item.findall("category")
            categories = (
                ",".join(filter(None, (_text_or_none(c) for c in categories_el)))
                or None
            )

            items.append(
                {
                    "source": feed.source,
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "published_at": published_iso,
                    "categories": categories,
                }
            )
    return items


def _strip_html(text: str) -> str:
    """Minimal tag stripper. Doesn't try to be an HTML parser."""
    import re

    # Drop tags + HTML entities that survive stdlib html.unescape unchanged.
    no_tags = re.sub(r"<[^>]+>", " ", text)
    collapsed = re.sub(r"\s+", " ", no_tags)
    import html as _html

    return _html.unescape(collapsed).strip()


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------


def fetch_feed(feed: Feed, *, timeout: float = TIMEOUT_SECONDS) -> tuple[int, bytes | None, str | None]:
    """GET the feed URL. Returns (http_status, body_or_None, error_or_None)."""
    try:
        response = requests.get(
            feed.url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/atom+xml, application/xml, */*"},
        )
    except requests.RequestException as exc:
        return (-1, None, f"request error: {exc}")
    if response.status_code >= 400:
        return (response.status_code, None, f"HTTP {response.status_code}")
    return (response.status_code, response.content, None)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def ingest_all(
    feeds: Iterable[Feed] = FEEDS,
    db_path: Path | str = DEFAULT_DB_PATH,
    dry_run: bool = False,
) -> list[dict]:
    """Pull every feed once, insert new headlines, return per-feed stats."""
    init_db(db_path)
    results: list[dict] = []

    for feed in feeds:
        t0 = time.time()
        status, body, error = fetch_feed(feed)
        if error or body is None:
            if not dry_run:
                with connect(db_path) as conn:
                    record_ingest(
                        conn, feed.source, feed.url, 0, 0, status, error
                    )
            results.append(
                {
                    "source": feed.source,
                    "feed": feed.url,
                    "status": status,
                    "found": 0,
                    "inserted": 0,
                    "error": error,
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            continue

        try:
            items = parse_feed(body, feed)
        except ValueError as exc:
            if not dry_run:
                with connect(db_path) as conn:
                    record_ingest(conn, feed.source, feed.url, 0, 0, status, str(exc))
            results.append(
                {
                    "source": feed.source,
                    "feed": feed.url,
                    "status": status,
                    "found": 0,
                    "inserted": 0,
                    "error": str(exc),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            continue

        inserted = 0
        if not dry_run:
            with connect(db_path) as conn:
                inserted = insert_headlines(conn, items)
                record_ingest(
                    conn, feed.source, feed.url, len(items), inserted, status, None
                )
        results.append(
            {
                "source": feed.source,
                "feed": feed.url,
                "status": status,
                "found": len(items),
                "inserted": inserted,
                "error": None,
                "elapsed_s": round(time.time() - t0, 2),
            }
        )
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Pull news RSS feeds into the local store.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse feeds and print counts without writing to the DB.",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="Limit to a single source (repeatable). Defaults to all.",
    )
    args = parser.parse_args()

    selected = FEEDS
    if args.source:
        wanted = {s.lower() for s in args.source}
        selected = tuple(f for f in FEEDS if f.source in wanted)
        if not selected:
            print(f"# no feeds match --source {args.source}", file=sys.stderr)
            sys.exit(2)

    results = ingest_all(selected, db_path=args.db, dry_run=args.dry_run)
    total_found = sum(r["found"] for r in results)
    total_inserted = sum(r["inserted"] for r in results)
    print(f"# fetched {len(results)} feed(s); found {total_found}, inserted {total_inserted}")
    for r in results:
        flag = "DRY" if args.dry_run else "OK "
        if r["error"]:
            flag = "ERR"
        print(
            f"  [{flag}] {r['source']:<15s} status={r['status']} "
            f"found={r['found']:>3d} inserted={r['inserted']:>3d} "
            f"({r['elapsed_s']}s){' — ' + r['error'] if r['error'] else ''}"
        )

    if not args.dry_run:
        with connect(args.db) as conn:
            st = stats(conn)
        print(f"# store totals: {st['total']} headlines "
              f"(oldest={st['oldest']}, newest={st['newest']})")
        for src, n in st["per_source"].items():
            print(f"    {src:<15s} {n}")


if __name__ == "__main__":
    _cli()
