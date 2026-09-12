"""Publish discoverable site assets without changing forecasts or owner tokens."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://etherforecast.live"
ASSETS = (
    "index.html", "robots.txt", "sitemap.xml", "site-info.css", "privacy.html",
    "ko/index.html", "social-preview.svg", "social-preview.png",
    "events.js", "event_diagnostics.js", "brand-status.js", "brand-status.css", "site_status.json",
    "assets/etherforecast-logo.svg", "assets/etherforecast-poster.svg",
)
HORIZONS = (6, 24, 72, 168, 336, 720)


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return result.astimezone(timezone.utc)


def date_label(value):
    return timestamp(value).strftime("%Y-%m-%d %H:%M UTC")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def valid_record(record, payload):
    try:
        if not isinstance(record, dict):
            return False
        hours = record["horizon_seconds"] / 3600
        p, q = record["terminal_down_flat_up"], record["price_quantiles"]
        slot = timestamp(payload["expected_slot"])
        issued, start, end = (timestamp(record[k]) for k in ("issued_at", "window_start", "target_end"))
        return (
            hours in HORIZONS and isinstance(record.get("forecast_id"), str) and bool(record["forecast_id"])
            and isinstance(p, list) and len(p) == 3
            and all(finite(v) and 0 <= v <= 1 for v in p) and math.isclose(sum(p), 1, abs_tol=1e-6)
            and isinstance(q, list) and len(q) == 3
            and all(finite(v) and v > 0 for v in q) and q == sorted(q)
            and finite(record.get("reference_price")) and record["reference_price"] > 0
            and all(finite(record.get(k)) for k in ("lower_barrier_price", "upper_barrier_price"))
            and 0 < record["lower_barrier_price"] < record["reference_price"] < record["upper_barrier_price"]
            and timestamp(record["input_cutoff"]) == slot == timestamp(record["slot"])
            and slot <= timestamp(record["available_at"]) <= issued < slot + timedelta(minutes=55)
            and issued <= timestamp(payload["generated_at"]) + timedelta(minutes=2)
            and start == slot + timedelta(hours=1) and issued < start
            and end - start == timedelta(hours=hours)
        )
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        return False


def render_snapshot(payload, now=None):
    """Render only original published records; an old snapshot is never called live."""
    now = now or datetime.now(timezone.utc)
    if not isinstance(payload, dict):
        payload = {}
    usable = False
    source_date = "Not available"
    state = "Forecast data is unavailable. Open the dashboard to check the latest update."
    try:
        generated = timestamp(payload["generated_at"])
        slot = timestamp(payload["expected_slot"])
        source_date = date_label(payload["generated_at"])
        usable = (payload.get("schema_version") == 1 and payload.get("status") == "ready"
                  and isinstance(payload.get("release_id"), str) and bool(payload["release_id"])
                  and slot == slot.replace(minute=0, second=0, microsecond=0)
                  and slot <= generated <= now + timedelta(minutes=2))
        if usable:
            state = "Dated publication snapshot. Check its timestamp before using these research forecasts."
            if now - min(generated, slot) > timedelta(minutes=100):
                state = "This publication snapshot is older than 100 minutes. These are archived values, not a current outlook."
        else:
            state = "This release is not ready or has invalid metadata. Forecast values are unavailable."
    except (KeyError, ValueError, TypeError, AttributeError, OverflowError):
        pass
    records = payload.get("current", [])
    if not isinstance(records, list):
        records = []
    rows = []
    for hours in HORIZONS:
        label = f"{hours} hours" if hours < 24 else f"{hours // 24} day" + ("s" if hours > 24 else "")
        matches = [r for r in records if isinstance(r, dict) and r.get("horizon_seconds") == hours * 3600]
        if not usable or len(matches) != 1 or not valid_record(matches[0], payload):
            rows.append(f'<tr><th scope="row">{label}</th><td colspan="8">Unavailable in this release</td></tr>')
            continue
        r = matches[0]
        q, p = r["price_quantiles"], r["terminal_down_flat_up"]
        money = lambda v: f"${v:,.2f}"
        rows.append(
            f'<tr data-forecast-id="{escape(r["forecast_id"], quote=True)}"><th scope="row">{label}</th>'
            f'<td>{date_label(r["issued_at"])}</td><td>{date_label(r["target_end"])}</td>'
            f'<td>{money(r["reference_price"])}</td><td>{money(q[1])}</td>'
            f'<td>{money(q[0])} – {money(q[2])}</td>'
            f'<td>{p[0]:.1%}</td><td>{p[1]:.1%}</td><td>{p[2]:.1%}</td></tr>'
        )
    release = escape(str(payload.get("release_id", "Not available")))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ethereum Forecast Snapshot: 6 Hours to 30 Days · ETH Forecast</title>
<meta name="description" content="Dated ETH research forecasts with original issue times, price ranges and probabilities across six windows. Check the publication time and compare with actual outcomes.">
<link rel="canonical" href="{BASE}/outlook.html"><link rel="stylesheet" href="/site-info.css">
<meta property="og:type" content="website"><meta property="og:title" content="Ethereum forecast · published research snapshot">
<meta property="og:description" content="Six forecast windows, original timestamps, ranges and probabilities. Research beta.">
<meta property="og:url" content="{BASE}/outlook.html"><meta property="og:image" content="{BASE}/social-preview.png">
<meta name="twitter:card" content="summary_large_image"></head>
<body><main class="guide"><nav aria-label="Main navigation"><a href="/">ETH Forecast dashboard</a><a href="/ko/" lang="ko">한국어 안내</a><a href="/#donation-panel">Support</a></nav>
<p class="eyebrow">Research beta · published records</p><h1>Ethereum forecast snapshot</h1>
<p class="lead">Six windows, with the original forecast dates and estimates. This page is regenerated from the same published release used by the dashboard.</p>
<section class="notice"><strong>Source release: {escape(source_date)}</strong><p>{escape(state)}</p>
<p>These values describe that dated release and may have expired since publication. <a href="/">Check the dashboard for the latest availability and performance.</a></p></section>
<div class="table-scroll" tabindex="0" role="region" aria-label="Dated Ethereum forecast table"><table>
<caption>ETH/USD research forecasts — values rounded for display</caption>
<thead><tr><th scope="col">Window</th><th scope="col">Issued</th><th scope="col">Target end</th><th scope="col">Reference</th><th scope="col">Median estimate</th><th scope="col">Intended 80% range</th><th scope="col">Lower</th><th scope="col">Middle</th><th scope="col">Higher</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<section><h2>What these estimates mean</h2><p>Lower, middle and higher refer to the final price relative to the model's lower and upper price barriers. They are not the probability of touching a level during the window. The middle category spans the two barriers; it does not mean exactly zero return. See the selected window on the dashboard for its boundaries.</p>
<p>The intended 80% range is a model estimate, not guaranteed coverage. A forecast is scored only after its target ends. Historical backtests and results from forecasts actually published beforehand are presented separately.</p>
<p>Outperformance over a no-change forecast is not assumed. Read the <a href="/#performance">measured errors and limitations</a> before drawing conclusions. This is statistical research, not investment advice.</p></section>
<section><h2>Keep this research available</h2><p>Voluntary support helps with data collection, hosting and model evaluation. Forecast access does not require a donation.</p><a class="button" href="/#donation-panel">Support the project</a></section>
<details><summary>Source release identifier</summary><p class="identifier">{release}</p><a href="/signals.json">Published source JSON</a></details>
<footer><a href="https://etherforecast.live/privacy.html" rel="privacy-policy">Privacy Policy</a> · <a href="https://etherforecast.live/privacy.html#korean" lang="ko">개인정보처리방침</a></footer>
</main></body></html>\n'''


def publish(target, source=ROOT, stage=False):
    target, source = Path(target).resolve(), Path(source).resolve()
    public = target / "forecast_site/public"
    for name in ASSETS:
        src, dst = source / "forecast_site/public" / name, public / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    src, dst = source / "forecast_site/vercel.json", target / "forecast_site/vercel.json"
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    try:
        payload = json.loads((public / "signals.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    (public / "outlook.html").write_text(render_snapshot(payload), encoding="utf-8")
    paths = [f"forecast_site/public/{name}" for name in (*ASSETS, "outlook.html")]
    paths.append("forecast_site/vercel.json")
    if stage:
        subprocess.run(["git", "-C", str(target), "add", "-f", "--", *paths], check=True)
    print("Search assets published; forecast JSON and existing verification files retained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True, help="Target repository checkout")
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    publish(args.target, stage=args.stage)
