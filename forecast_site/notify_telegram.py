"""Send a Telegram notification summarising the latest ETH forecast run.

Designed to be appended to the daily pipeline as a best-effort, non-fatal
step: if the token or chat id is missing, we log a message and exit 0 so the
cron doesn't turn red just because someone hasn't set up Telegram yet.

Usage:
    # Requires env vars:
    #   TELEGRAM_BOT_TOKEN  -- from @BotFather
    #   TELEGRAM_CHAT_ID    -- your numeric user id or a group / channel id
    export TELEGRAM_BOT_TOKEN=123456:ABC...
    export TELEGRAM_CHAT_ID=12345678
    python -m forecast_site.notify_telegram

    # Dry run — prints the message instead of posting:
    python -m forecast_site.notify_telegram --dry-run

Message format (Markdown):
    📈 *ETH forecast · 2026-04-20*
    Reference close: $2,420 (2026-04-19)

    *h=7* target 2026-04-27
      UP · prob 58% · predicted $2,449 (+1.2%)
      bear $2,380  base $2,449  bull $2,520

    *h=30* target 2026-05-20
      UP · prob 62% · predicted $2,511 (+3.8%)
      bear $2,360  base $2,511  bull $2,675

    Track record (last 90d):
      h=7  · dir 67%  brier 0.188  n=12
      h=30 · dir 55%  brier 0.231  n=8

    _Not financial advice._
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_PUBLIC_DIR = Path(__file__).parent / "public"
TELEGRAM_API_TMPL = "https://api.telegram.org/bot{token}/sendMessage"


def _fmt_usd(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_pct(v, decimals: int = 1) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_date(iso: str | None) -> str:
    if not iso:
        return "—"
    return iso[:10]


def _build_message(latest: dict, accuracy: dict | None) -> str:
    run = latest.get("run") or {}
    if not run:
        return "📉 *ETH forecast* — no runs persisted yet."

    lines: list[str] = []
    lines.append(f"📈 *ETH forecast · {_fmt_date(run.get('run_timestamp_utc'))}*")
    lines.append(
        f"Reference close: {_fmt_usd(run.get('reference_price'))}"
        f" ({_fmt_date(run.get('reference_price_timestamp_utc'))})"
    )
    lines.append("")

    horizons = latest.get("horizons") or {}
    for h in sorted(horizons.keys(), key=lambda x: int(x)):
        f = horizons[h]
        direction = (f.get("hybrid_predicted_direction")
                     or f.get("classification_predicted_direction") or "FLAT").upper()
        prob_up = _fmt_pct(f.get("classification_probability_up"), decimals=0)
        predicted = f.get("active_predicted_close") or f.get("regression_predicted_close")
        delta = f.get("regression_predicted_return")
        target = _fmt_date(f.get("forecast_target_timestamp_utc"))

        lines.append(f"*h={h}* target {target}")
        lines.append(
            f"  {direction} · prob {prob_up} · predicted {_fmt_usd(predicted)} ({_fmt_pct(delta)})"
        )
        bear = f.get("active_bear_predicted_close")
        base = f.get("active_predicted_close")
        bull = f.get("active_bull_predicted_close")
        if bear or base or bull:
            lines.append(
                f"  bear {_fmt_usd(bear)}  base {_fmt_usd(base)}  bull {_fmt_usd(bull)}"
            )
        lines.append("")

    if accuracy and accuracy.get("horizons"):
        lines.append("Track record (last 90d):")
        for h in sorted(accuracy["horizons"].keys(), key=lambda x: int(x)):
            window = accuracy["horizons"][h].get("90") or accuracy["horizons"][h].get("all")
            if not window:
                continue
            dir_acc = _fmt_pct(window.get("direction_accuracy"), decimals=0)
            brier = window.get("brier_score")
            brier_s = f"{brier:.3f}" if brier is not None else "—"
            n = window.get("resolved_count") or 0
            lines.append(f"  h={h}  · dir {dir_acc}  brier {brier_s}  n={n}")
        lines.append("")

    lines.append("_Not financial advice._")
    return "\n".join(lines)


def _post(token: str, chat_id: str, text: str) -> dict:
    url = TELEGRAM_API_TMPL.format(token=token)
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram API HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR,
                        help="Directory containing latest.json / accuracy.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the message without posting to Telegram")
    parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    args = parser.parse_args()

    latest_path = args.public_dir / "latest.json"
    accuracy_path = args.public_dir / "accuracy.json"
    if not latest_path.exists():
        print(f"[notify_telegram] SKIP: {latest_path} missing "
              f"(run forecast_site.export_json first)")
        return 0

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    accuracy = (json.loads(accuracy_path.read_text(encoding="utf-8"))
                if accuracy_path.exists() else None)
    message = _build_message(latest, accuracy)

    if args.dry_run:
        # Emojis + unicode in the message can trip cp949 stdout on Windows.
        # Reconfigure the buffer to UTF-8 so the preview always prints cleanly.
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
        except Exception:
            pass
        print("--- Telegram message (dry run) ---")
        print(message)
        return 0

    if not args.token or not args.chat_id:
        print("[notify_telegram] SKIP: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
              "Add them to GitHub secrets / local env to enable alerts.")
        return 0

    try:
        result = _post(args.token, args.chat_id, message)
    except Exception as exc:  # noqa: BLE001
        # Best-effort: never fail the cron because Telegram is down.
        print(f"[notify_telegram] WARN: {exc}", file=sys.stderr)
        return 0

    if result.get("ok"):
        msg_id = (result.get("result") or {}).get("message_id")
        print(f"[notify_telegram] sent (message_id={msg_id})")
    else:
        # API returned something but not ok — log and move on.
        print(f"[notify_telegram] WARN: non-ok response: {result}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
