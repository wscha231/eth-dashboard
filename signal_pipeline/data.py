"""Closed hourly USD bars, append-only source revisions and reproducible features."""
from datetime import timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .protocol import PRODUCTS, SOURCE, WARMUP_HOURS


def utc(value=None):
    value = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def fetch(url, deadline):
    """One documented host. Denied requests stop; bounded retries respect 429."""
    for attempt in range(3):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("collection budget exhausted")
        try:
            with urlopen(Request(url, headers={"User-Agent": "etherforecast-hourly/1.0"}),
                         timeout=min(15, remaining)) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("oversized candle response")
            return raw
        except HTTPError as exc:
            if exc.code in (401, 403, 451) or exc.code not in (429, 500, 502, 503, 504):
                raise
            retry = exc.headers.get("Retry-After", str(2 ** attempt))
            try:
                delay = float(retry)
            except ValueError:
                delay = max(0., (utc(retry)-utc()).total_seconds())
        except (TimeoutError, OSError):
            delay = 2 ** attempt
        if attempt == 2 or delay > deadline-time.monotonic() or delay > 30:
            raise TimeoutError("source unavailable within retry budget")
        time.sleep(delay)


def parse_candles(raw, start, end, observed_at):
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("expected candle list")
    if any(not isinstance(row, list) or len(row) != 6 for row in payload):
        raise ValueError("expected time, low, high, open, close, volume")
    frame = pd.DataFrame(payload, columns=["timestamp", "low", "high", "open", "close", "volume"])
    if frame.empty:
        return frame
    numeric = frame.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite candle")
    if (frame.timestamp % 3600 != 0).any():
        raise ValueError("timestamps must be aligned UTC seconds, not ms/us")
    frame["open_time"] = pd.to_datetime(frame.timestamp, unit="s", utc=True)
    frame["close_time"] = frame.open_time + pd.Timedelta(hours=1)
    if frame.open_time.duplicated().any():
        raise ValueError("duplicate candle")
    if ((frame[["open", "high", "low", "close"]] <= 0).any().any()
            or (frame.volume < 0).any()
            or (frame.high < frame[["open", "close", "low"]].max(axis=1)).any()
            or (frame.low > frame[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("invalid OHLCV bounds")
    return frame.loc[(frame.open_time >= utc(start)) & (frame.open_time < utc(end)) &
                     (frame.close_time <= utc(observed_at))].sort_values("open_time")


def connect(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "observations.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS bars (
        product TEXT, open_time TEXT, observed_at TEXT, revision INTEGER,
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        content_hash TEXT, raw_hash TEXT, PRIMARY KEY(product,open_time,revision))""")
    con.execute("CREATE INDEX IF NOT EXISTS bars_lookup ON bars(product,open_time,revision DESC)")
    con.execute("""CREATE TABLE IF NOT EXISTS downloads (
        url TEXT, raw_hash TEXT, observed_at TEXT, rows INTEGER,
        PRIMARY KEY(url,raw_hash))""")
    return con


def ingest(con, root, product, raw, start, end, observed_at):
    frame = parse_candles(raw, start, end, observed_at)
    raw_hash = hashlib.sha256(raw).hexdigest()
    raw_dir = Path(root)/"raw"; raw_dir.mkdir(exist_ok=True)
    target = raw_dir/f"{raw_hash}.json.gz"
    if not target.exists():
        target.write_bytes(gzip.compress(raw, mtime=0))
    for row in frame.itertuples():
        values = [float(getattr(row, key)) for key in ("open", "high", "low", "close", "volume")]
        content_hash = hashlib.sha256(json.dumps(values).encode()).hexdigest()
        t = row.open_time.isoformat()
        prior = con.execute("SELECT revision,content_hash FROM bars WHERE product=? AND open_time=? ORDER BY revision DESC LIMIT 1", (product, t)).fetchone()
        if prior and prior[1] == content_hash:
            continue
        revision = prior[0]+1 if prior else 1
        con.execute("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (product, t, utc(observed_at).isoformat(), revision, *values, content_hash, raw_hash))
    return len(frame), raw_hash


def read_bars(root, *, as_of=None):
    with connect(root) as con:
        condition, params = ("WHERE observed_at<=?", (utc(as_of).isoformat(),)) if as_of is not None else ("", ())
        frame = pd.read_sql_query(f"""SELECT * FROM (SELECT *,ROW_NUMBER() OVER
            (PARTITION BY product,open_time ORDER BY revision DESC) AS rn FROM bars {condition})
            WHERE rn=1 ORDER BY open_time,product""", con, params=params)
    for col in ("open_time", "observed_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["close_time"] = frame.open_time + pd.Timedelta(hours=1)
    return frame.drop(columns=["rn"])


def collect(root, *, start="2017-01-01", now=None, budget_seconds=300, max_requests=800, fetcher=fetch):
    started = time.monotonic(); deadline = started + budget_seconds
    current = utc(now); end = current.floor("h")
    errors = []; requests = 0; stopped = False
    with connect(root) as con:
        for product in PRODUCTS:
            cached = {utc(r[0]) for r in con.execute("SELECT DISTINCT open_time FROM bars WHERE product=?", (product,))}
            expected = pd.date_range(utc(start), end-pd.Timedelta(hours=1), freq="h")
            # Latest closed bars first, then historical gaps. Revisit last six hours for corrections.
            missing = set(expected)-cached
            missing.update(pd.date_range(max(utc(start), end-pd.Timedelta(hours=6)), end-pd.Timedelta(hours=1), freq="h"))
            while missing and time.monotonic() < deadline and requests < max_requests:
                upper = max(missing)+pd.Timedelta(hours=1)
                lower = max(utc(start), min(missing), upper-pd.Timedelta(hours=299))
                url = f"https://api.exchange.coinbase.com/products/{product}/candles?" + urlencode({
                    "granularity": 3600, "start": lower.isoformat(), "end": upper.isoformat()})
                try:
                    requests += 1
                    raw = fetcher(url, deadline)
                    observed = utc()  # actual receipt time, including backfilled data
                    with con:
                        rows, raw_hash = ingest(con, root, product, raw, lower, upper, observed)
                        con.execute("INSERT OR IGNORE INTO downloads VALUES (?,?,?,?)", (url, raw_hash, observed.isoformat(), rows))
                    # An empty response is a recorded gap; don't hammer it in this run.
                    missing.difference_update(pd.date_range(lower, upper-pd.Timedelta(hours=1), freq="h"))
                except Exception as exc:
                    errors.append({"product": product, "error": type(exc).__name__, "detail": str(exc)[:180]})
                    stopped = True
                    break
            if stopped:
                break
    bars = read_bars(root)
    latest = {p: (bars.loc[bars["product"].eq(p), "close_time"].max().isoformat() if (bars["product"] == p).any() else None) for p in PRODUCTS}
    report = {"source": SOURCE, "instrument": "ETH-USD", "generated_at": utc().isoformat(),
              "expected_cutoff": end.isoformat(), "latest": latest, "rows": len(bars),
              "requests": requests, "errors": errors, "runtime_seconds": time.monotonic()-started,
              "ready": all(v == end.isoformat() for v in latest.values()),
              "history": "observed now; historical reconstruction, not historical receipt vintages"}
    if len(bars):
        # Portable normalized snapshot; source revisions and raw evidence remain separate.
        bars.to_parquet(Path(root)/"hourly.parquet", index=False)
    (Path(root)/"source_status.json").write_text(json.dumps(report, indent=2))
    return bars, report


def build_features(bars):
    streams = {}
    for product in PRODUCTS:
        f = bars.loc[bars["product"].eq(product)].set_index("close_time").sort_index()
        if f.index.has_duplicates:
            raise ValueError("duplicate source vintage")
        streams[product] = f
    if any(f.empty for f in streams.values()):
        return pd.DataFrame()
    lo = min(f.index.min() for f in streams.values()); hi = max(f.index.max() for f in streams.values())
    index = pd.date_range(lo, hi, freq="h", tz="UTC")
    eth, btc = [streams[p].reindex(index) for p in PRODUCTS]
    out = pd.DataFrame(index=index)
    for prefix, f in (("eth", eth), ("btc", btc)):
        logp = np.log(f.close); ret = logp.diff()
        for lag in (1, 6, 24, 72, 168):
            out[f"{prefix}_ret_{lag}"] = logp.diff(lag)
        for n in (24, 168, WARMUP_HOURS):
            out[f"{prefix}_vol_{n}"] = ret.rolling(n, min_periods=n).std()
            out[f"{prefix}_trend_{n}"] = logp-logp.rolling(n, min_periods=n).mean()
        out[f"{prefix}_range"] = np.log(f.high/f.low)
        out[f"{prefix}_volume_z"] = (np.log1p(f.volume)-np.log1p(f.volume).rolling(168).mean()) / np.log1p(f.volume).rolling(168).std().replace(0, np.nan)
        out[f"{prefix}_drawdown"] = logp-logp.rolling(168).max()
    out["eth_btc_ret_24"] = out.eth_ret_24-out.btc_ret_24
    out["eth_btc_corr"] = np.log(eth.close).diff().rolling(168).corr(np.log(btc.close).diff())
    out["hour_sin"] = np.sin(index.hour*2*np.pi/24)
    out["hour_cos"] = np.cos(index.hour*2*np.pi/24)
    valid = eth.close.notna() & btc.close.notna()
    valid = valid.rolling(WARMUP_HOURS+1, min_periods=WARMUP_HOURS+1).sum().eq(WARMUP_HOURS+1)
    out.loc[~valid] = np.nan
    out["reference_price"] = eth.close
    out["sigma"] = out.eth_vol_720
    # Actual receipt cutoff for the entire required window, including backfill/corrections.
    obs = pd.concat([eth.observed_at, btc.observed_at], axis=1)
    latest_obs = obs.apply(lambda s: s.astype("int64")).max(axis=1)
    latest_obs = latest_obs.rolling(WARMUP_HOURS+1).max()
    out["available_at"] = pd.to_datetime(latest_obs, utc=True)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame):
    return [c for c in frame if c not in {"reference_price", "sigma", "available_at"}]
