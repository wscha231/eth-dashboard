from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Any

SOURCE_ID = "bybit_v5_all_liquidation_ethusdt"
SOURCE_NAME = "Bybit V5 all liquidation public stream — ETHUSDT"
WS_URL = "wss://stream.bybit.com/v5/public/linear"
TOPIC = "allLiquidation.ETHUSDT"
SYMBOL = "ETHUSDT"
LICENSE_STATUS = "rights_unreviewed_public_stream"
PUBLIC_REDISTRIBUTION = "blocked_pending_rights_review"
MODEL_USE = "blocked_until_rights_prospective_continuity_and_preregistered_ablation_pass"
SITE_USE = "blocked"
HISTORY_MODE = "prospective_receipts_only"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _ms_iso(value: Any) -> str:
    try:
        milliseconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("liquidation timestamp must be integer milliseconds") from exc
    if milliseconds <= 0:
        raise ValueError("liquidation timestamp must be positive")
    return datetime.fromtimestamp(milliseconds / 1000, tz=timezone.utc).isoformat()


def _positive_decimal(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _event_id(event: dict[str, Any]) -> str:
    identity = {
        "source_id": SOURCE_ID,
        "T": event["T"],
        "s": event["s"],
        "S": event["S"],
        "v": str(event["v"]),
        "p": str(event["p"]),
    }
    return hashlib.sha256(_canonical(identity)).hexdigest()


def parse_bybit_message(message: dict[str, Any], *, received_at: str) -> tuple[bool, list[dict[str, Any]]]:
    """Return (subscription_acknowledged, normalized liquidation rows).

    The position-side mapping follows Bybit's documented stream semantics:
    source side Buy means a long position was liquidated; Sell means a short
    position was liquidated.  No directional trading interpretation is added.
    """
    if message.get("op") == "subscribe":
        return bool(message.get("success")), []
    if message.get("topic") != TOPIC:
        return False, []
    data = message.get("data")
    if not isinstance(data, list):
        raise ValueError("Bybit liquidation data must be a list")
    stream_generated_at = _ms_iso(message.get("ts"))
    rows: list[dict[str, Any]] = []
    for event in data:
        if not isinstance(event, dict):
            raise ValueError("liquidation event must be an object")
        if event.get("s") != SYMBOL:
            raise ValueError("unexpected liquidation symbol")
        side = str(event.get("S") or "")
        if side not in {"Buy", "Sell"}:
            raise ValueError("unexpected liquidation side")
        size = _positive_decimal(event.get("v"), "executed size")
        price = _positive_decimal(event.get("p"), "bankruptcy price")
        row = {
            "event_id": _event_id(event),
            "source_id": SOURCE_ID,
            "symbol": SYMBOL,
            "event_time": _ms_iso(event.get("T")),
            "stream_generated_at": stream_generated_at,
            "received_at": received_at,
            "available_at": received_at,
            "source_side": side,
            "liquidated_position": "long" if side == "Buy" else "short",
            "size_eth": format(size, "f"),
            "bankruptcy_price_usdt": format(price, "f"),
            "notional_usdt": format(size * price, "f"),
            "notional_is_derived": True,
        }
        rows.append(row)
    return False, rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _minute_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for row in events:
        minute = datetime.fromisoformat(row["event_time"]).astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
        bucket = buckets.setdefault(minute, {
            "minute_utc": minute,
            "event_count": 0,
            "long_liquidation_count": 0,
            "short_liquidation_count": 0,
            "long_liquidation_notional_usdt": Decimal(0),
            "short_liquidation_notional_usdt": Decimal(0),
            "total_liquidation_notional_usdt": Decimal(0),
        })
        value = Decimal(row["notional_usdt"])
        bucket["event_count"] += 1
        bucket["total_liquidation_notional_usdt"] += value
        position = row["liquidated_position"]
        bucket[f"{position}_liquidation_count"] += 1
        bucket[f"{position}_liquidation_notional_usdt"] += value
    output: list[dict[str, Any]] = []
    for minute in sorted(buckets):
        row = buckets[minute]
        output.append({key: (format(value, "f") if isinstance(value, Decimal) else value) for key, value in row.items()})
    return output


def persist_session(root: str | Path, *, events: list[dict[str, Any]], session: dict[str, Any]) -> dict[str, Any]:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    ledger = base / "eth_liquidation_events.jsonl"
    prior = _read_jsonl(ledger)
    by_id = {str(row["event_id"]): row for row in prior}
    for row in events:
        by_id.setdefault(str(row["event_id"]), row)
    merged = sorted(by_id.values(), key=lambda row: (row["event_time"], row["event_id"]))
    with ledger.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _write_csv(base / "eth_liquidation_events.csv", merged)
    _write_csv(base / "eth_liquidation_1m.csv", _minute_rows(merged))

    sessions_path = base / "capture_sessions.jsonl"
    with sessions_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(session, sort_keys=True, separators=(",", ":")) + "\n")
    sessions = _read_jsonl(sessions_path)

    total_notional = sum((Decimal(row["notional_usdt"]) for row in merged), Decimal(0))
    long_notional = sum((Decimal(row["notional_usdt"]) for row in merged if row["liquidated_position"] == "long"), Decimal(0))
    short_notional = total_notional - long_notional
    report = {
        "schema": 1,
        "generated_at_utc": session["ended_at"],
        "status": "research_only_rights_unreviewed",
        "source_id": SOURCE_ID,
        "source_name": SOURCE_NAME,
        "source_url": WS_URL,
        "topic": TOPIC,
        "license_status": LICENSE_STATUS,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "history_mode": HISTORY_MODE,
        "historical_backfill": "not_eligible_no_market_liquidation_history_endpoint_used",
        "available_at_policy": "actual EtherForecast WebSocket receipt time",
        "capture_sessions": len(sessions),
        "complete_capture_sessions": sum(bool(x.get("complete")) for x in sessions),
        "subscription_acknowledged": bool(session.get("subscription_acknowledged")),
        "last_session_complete": bool(session.get("complete")),
        "last_session_duration_seconds": session.get("actual_duration_seconds"),
        "last_session_events": session.get("event_count"),
        "last_session_disconnects": session.get("disconnects"),
        "unique_event_rows": len(merged),
        "total_liquidation_notional_usdt": format(total_notional, "f"),
        "long_liquidation_notional_usdt": format(long_notional, "f"),
        "short_liquidation_notional_usdt": format(short_notional, "f"),
        "latest_event_time": merged[-1]["event_time"] if merged else None,
    }
    (base / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def collect_bybit_liquidations(root: str | Path, *, duration_seconds: int, websocket_url: str = WS_URL) -> dict[str, Any]:
    """Collect prospective liquidation receipts for a bounded observation window.

    This intentionally has no historical fallback.  Missing coverage remains a
    recorded gap rather than being synthesized or inferred.
    """
    if duration_seconds < 5 or duration_seconds > 4 * 60 * 60:
        raise ValueError("duration_seconds must be between 5 seconds and 4 hours")
    try:
        import websocket  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in the workflow environment
        raise RuntimeError("websocket-client is required for live liquidation collection") from exc

    started = utc_now()
    deadline = time.monotonic() + duration_seconds
    rows: list[dict[str, Any]] = []
    acknowledged = False
    disconnects = 0
    parse_errors = 0
    connections = 0
    reached_deadline = False

    while time.monotonic() < deadline:
        ws = None
        try:
            ws = websocket.create_connection(websocket_url, timeout=8)
            connections += 1
            ws.send(json.dumps({"op": "subscribe", "args": [TOPIC]}, separators=(",", ":")))
            last_ping = time.monotonic()
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                ws.settimeout(max(1.0, min(5.0, remaining)))
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    if time.monotonic() - last_ping >= 20:
                        ws.send(json.dumps({"op": "ping"}, separators=(",", ":")))
                        last_ping = time.monotonic()
                    continue
                if not raw:
                    continue
                received = _iso(utc_now())
                try:
                    message = json.loads(raw)
                    ack, parsed = parse_bybit_message(message, received_at=received)
                    acknowledged = acknowledged or ack
                    rows.extend(parsed)
                except (json.JSONDecodeError, ValueError, TypeError):
                    parse_errors += 1
            reached_deadline = True
        except Exception:
            disconnects += 1
            if time.monotonic() >= deadline:
                reached_deadline = True
                break
            time.sleep(min(3.0, max(0.0, deadline - time.monotonic())))
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
        if reached_deadline:
            break

    ended = utc_now()
    actual = max(0.0, (ended - started).total_seconds())
    session = {
        "schema": 1,
        "source_id": SOURCE_ID,
        "topic": TOPIC,
        "started_at": _iso(started),
        "ended_at": _iso(ended),
        "planned_duration_seconds": duration_seconds,
        "actual_duration_seconds": round(actual, 3),
        "subscription_acknowledged": acknowledged,
        "complete": bool(acknowledged and reached_deadline and actual >= duration_seconds * 0.90),
        "connections": connections,
        "disconnects": disconnects,
        "parse_errors": parse_errors,
        "event_count": len(rows),
    }
    return persist_session(root, events=rows, session=session)
