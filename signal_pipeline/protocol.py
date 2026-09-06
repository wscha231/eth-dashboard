"""Frozen event definitions; changing these creates a new experiment family."""
import hashlib
import json
from pathlib import Path

VERSION = "hourly_events_v1"
HORIZONS = {6: .015, 24: .03, 72: .05, 168: .07, 336: .10, 720: .15}
DEFAULT_HORIZONS = (24, 72, 168)
SOURCE = "coinbase_exchange"
PRODUCTS = ("ETH-USD", "BTC-USD")
WARMUP_HOURS = 720
SPEC = {
    "version": VERSION, "source": SOURCE, "instrument": "ETH-USD",
    "quote": "USD", "horizon_hours_and_floor": HORIZONS,
    "bar_seconds": 3600, "feature_warmup_hours": WARMUP_HOURS,
    "input": "closed ETH and BTC hourly bars; no filling missing bars",
    "barrier": "max(log(1+floor), trailing_720h_logreturn_std*sqrt(h))",
    "reference": "ETH close at input_cutoff, never rebased after issue",
    "window": "first complete UTC hour after issuance slot, then h hours",
    "terminal": "down/flat/up at window_end using log-symmetric barrier",
    "path": "independent high/low barrier hits; both can occur; no first-hit order",
    "training_stride_hours": 6, "replay_stride_hours": 24,
    "train_days": 730, "validation_days": 90, "embargo_hours": 1,
    "selection": "inner purged Brier; CatBoost/climatology blends 0,.5,1",
    "refit": "monthly; hourly inference never fits",
    "historical_status": "reconstructed research; not actual issued vintages",
    "prospective_status": "shadow research until independent evidence supports promotion",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


PROTOCOL_HASH = digest(SPEC)


def runtime_hash():
    root = Path(__file__).parent
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes()
                                    for p in sorted(root.glob("*.py")))).hexdigest()
