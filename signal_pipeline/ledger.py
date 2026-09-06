"""Immutable actual issuance, append-only settlement revisions and publication outbox."""
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .data import utc
from .protocol import PROTOCOL_HASH, digest


def connect(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(Path(root)/"issued.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
      CREATE TABLE IF NOT EXISTS forecasts (
        forecast_id TEXT PRIMARY KEY, slot TEXT, horizon_seconds INTEGER,
        model_version TEXT, target_definition TEXT, issued_at TEXT,
        target_end TEXT, payload TEXT,
        UNIQUE(slot,horizon_seconds,model_version,target_definition));
      CREATE TABLE IF NOT EXISTS outcomes (
        forecast_id TEXT REFERENCES forecasts(forecast_id), revision INTEGER,
        evaluated_at TEXT, truth_hash TEXT, payload TEXT,
        PRIMARY KEY(forecast_id,revision), UNIQUE(forecast_id,truth_hash));
      CREATE TABLE IF NOT EXISTS outbox (
        forecast_id TEXT PRIMARY KEY REFERENCES forecasts(forecast_id), created_at TEXT);
      CREATE TABLE IF NOT EXISTS deliveries (
        forecast_id TEXT REFERENCES forecasts(forecast_id), verified_at TEXT,
        release_id TEXT, PRIMARY KEY(forecast_id,release_id));
      CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, started_at TEXT, status TEXT, payload TEXT);
    """)
    return con


def validate_forecast(record, now):
    now = utc(now); slot = utc(record["slot"])
    if slot != now.floor("h") or now >= slot+pd.Timedelta(minutes=55):
        raise ValueError("only the current hourly slot can be actually issued")
    if utc(record["input_cutoff"]) != slot or utc(record["available_at"]) > now:
        raise ValueError("stale input or future source receipt")
    if utc(record["window_start"]) != slot+pd.Timedelta(hours=1):
        raise ValueError("event window must start after actual issuance")
    if utc(record["training_target_end"]) >= slot or utc(record["validation_target_end"]) >= slot:
        raise ValueError("unmatured training/selection target")
    if (utc(record["target_end"])-utc(record["window_start"])).total_seconds() != record["horizon_seconds"]:
        raise ValueError("inconsistent horizon")
    p = np.asarray(record["terminal_down_flat_up"])
    hit = np.asarray([record["hit_up"], record["hit_down"]])
    q = np.asarray(record["price_quantiles"])
    if (not np.isfinite(np.r_[p,hit,q]).all() or np.any(p < 0) or np.any(p > 1)
            or not np.isclose(p.sum(), 1) or np.any(hit < 0) or np.any(hit > 1)
            or np.any(q <= 0) or np.any(np.diff(q) < 0)):
        raise ValueError("invalid probability or quantile output")


def issue(root, record, *, now=None):
    now = utc(now)
    key = {k: record[k] for k in ("slot", "horizon_seconds", "model_version")}
    key["target_definition"] = PROTOCOL_HASH
    forecast_id = digest(key)
    with connect(root) as con:
        con.execute("BEGIN IMMEDIATE")
        previous = con.execute("SELECT payload FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone()
        if previous:
            return json.loads(previous[0])
        validate_forecast(record, now)
        payload = {**record, **key, "forecast_id": forecast_id, "issued_at": now.isoformat(), "role": "shadow_research"}
        serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        con.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?)", (forecast_id, record["slot"],
                    record["horizon_seconds"], record["model_version"], PROTOCOL_HASH, now.isoformat(), record["target_end"], serialized))
        con.execute("INSERT INTO outbox VALUES (?,?)", (forecast_id, now.isoformat()))
    return payload


def settle(root, bars, *, now=None):
    now = utc(now)
    eth = bars.loc[bars["product"].eq("ETH-USD") & (bars.observed_at <= now)].set_index("open_time").sort_index()
    settled = 0
    with connect(root) as con:
        for forecast_id, raw in con.execute("SELECT forecast_id,payload FROM forecasts WHERE target_end<=?", (now.isoformat(),)).fetchall():
            forecast = json.loads(raw)
            start, end = utc(forecast["window_start"]), utc(forecast["target_end"])
            expected = pd.date_range(start, end-pd.Timedelta(hours=1), freq="h")
            window = eth.reindex(expected)
            if window[["open", "high", "low", "close"]].isna().any().any():
                continue  # a missing truth bar is pending, not a negative event
            ref, barrier = forecast["reference_price"], forecast["log_barrier"]
            actual = float(window.close.iloc[-1]); ret = float(np.log(actual/ref))
            up = window.high.ge(ref*np.exp(barrier)); down = window.low.le(ref*np.exp(-barrier))
            truth_hash = digest(window.content_hash.tolist())
            result = {"actual_price": actual, "return": ret, "up": int(up.any()), "down": int(down.any()),
                      "terminal": 2 if ret > barrier else 0 if ret < -barrier else 1,
                      "first_up_bar": up.index[up][0].isoformat() if up.any() else None,
                      "first_down_bar": down.index[down][0].isoformat() if down.any() else None,
                      "truth_available_at": window.observed_at.max().isoformat()}
            prior = con.execute("SELECT 1 FROM outcomes WHERE forecast_id=? AND truth_hash=?", (forecast_id, truth_hash)).fetchone()
            if prior:
                continue
            revision = con.execute("SELECT COALESCE(MAX(revision),0)+1 FROM outcomes WHERE forecast_id=?", (forecast_id,)).fetchone()[0]
            con.execute("INSERT INTO outcomes VALUES (?,?,?,?,?)", (forecast_id, revision, now.isoformat(), truth_hash, json.dumps(result)))
            settled += 1
    return settled


def history(root):
    with connect(root) as con:
        rows = con.execute("""SELECT f.payload,o.payload,o.revision FROM forecasts f LEFT JOIN outcomes o
            ON f.forecast_id=o.forecast_id AND o.revision=(SELECT MAX(revision) FROM outcomes WHERE forecast_id=f.forecast_id)
            ORDER BY f.issued_at,f.horizon_seconds""").fetchall()
    return [{**json.loads(f), "outcome": json.loads(o) if o else None, "truth_revision": revision} for f,o,revision in rows]


def mark_verified(root, forecast_ids, release_id):
    with connect(root) as con:
        for fid in forecast_ids:
            con.execute("INSERT OR IGNORE INTO deliveries VALUES (?,?,?)", (fid, utc().isoformat(), release_id))


def backup(root, destination):
    """SQLite online backups include WAL writes; copy-files alone is not a backup."""
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    for name in ("issued.db", "observations.db"):
        path = Path(root)/name
        if not path.exists():
            continue
        with sqlite3.connect(path) as src, sqlite3.connect(destination/name) as dest:
            src.backup(dest)
            if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("backup integrity check failed")
