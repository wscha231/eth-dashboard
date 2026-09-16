"""DOM-specific Farside Ethereum ETF table parser.

Farside's live page may wrap the actual table rows inside ancestor rows whose recursive
cell search sees thousands of descendants. This parser deliberately considers only direct
th/td children and recognizes a row as data only when its first direct cell is an exact
`DD Mon YYYY` date. Source/PIT/version semantics remain in eth_etf_flows.py.
"""
from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup
import pandas as pd

from .eth_etf_flows import NUMERIC_COLUMNS, TICKERS, _find_flow_table, parse_flow_cell

DATE_RE = re.compile(r"^\d{1,2} [A-Z][a-z]{2} \d{4}$")


def parse_farside_html(raw: bytes) -> pd.DataFrame:
    if not raw:
        raise ValueError("empty ETF HTML")
    soup = BeautifulSoup(raw, "html.parser")
    table = _find_flow_table(soup)
    dated: list[dict[str, Any]] = []
    expected = len(TICKERS) + 2

    for tr in table.find_all("tr"):
        direct = tr.find_all(["th", "td"], recursive=False)
        if not direct:
            continue
        cells = [cell.get_text(" ", strip=True) for cell in direct]
        if not cells or not DATE_RE.fullmatch(cells[0].strip()):
            continue
        if len(cells) != expected:
            raise ValueError(f"direct dated ETF row has {len(cells)} cells; expected {expected}")
        date = pd.to_datetime(cells[0], format="%d %b %Y", utc=True, errors="raise").floor("D")
        row: dict[str, Any] = {"date": date}
        for ticker, cell in zip(TICKERS, cells[1 : 1 + len(TICKERS)], strict=True):
            row[ticker.lower() + "_usd_m"] = parse_flow_cell(cell)
        row["total_usd_m"] = parse_flow_cell(cells[-1])
        dated.append(row)

    if not dated:
        raise ValueError("no direct dated Ethereum ETF rows found")
    frame = pd.DataFrame(dated).sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("duplicate Ethereum ETF event dates")
    if frame.iloc[0]["date"] != pd.Timestamp("2024-07-23", tz="UTC"):
        raise ValueError("unexpected first Ethereum ETF flow date")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Ethereum ETF dates are not monotonic")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(NUMERIC_COLUMNS)].isna().all(axis=1).any():
        raise ValueError("ETF row contains no numeric evidence")
    return frame
