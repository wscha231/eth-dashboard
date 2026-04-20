"""Populate forecast_site/predictions.db with a few synthetic runs + resolved
actuals so the static frontend (public/index.html) has something to display.

Runs:
  - 90 days ago  : synthetic forecast, target date already past  -> resolved.
  - 60 days ago  : synthetic forecast, target date already past  -> resolved.
  - 30 days ago  : synthetic forecast, h=7 target resolved, h=30 target unresolved.
  - today        : "current" forecast, both horizons unresolved (the hero card).

Usage:
    python -m forecast_site.seed_demo
    python -m forecast_site.seed_demo --wipe   # drop existing demo rows first
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402
from forecast_site.backfill_actuals import (  # noqa: E402
    _load_eth_close_lookup, refresh_accuracy_snapshots, resolve_pending_forecasts,
)
from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402
from forecast_site.persist_forecast import persist_forecast  # noqa: E402
from forecast_site.smoke_test import _synthetic_row  # noqa: E402

MASTER_CSV = PROJECT_ROOT / "lake" / "gold" / "eth_master_daily.csv"


def _seed_run(db_path: Path, master_csv: Path, days_ago: int, label: str,
              pred_return_7: float, pred_return_30: float,
              prob_up_7: float, prob_up_30: float) -> None:
    close_lookup = _load_eth_close_lookup(master_csv)
    today_utc = pd.Timestamp.now(tz="UTC").floor("D")
    input_ts = today_utc - pd.Timedelta(days=days_ago)
    if input_ts not in close_lookup.index:
        # Use the latest available close before that date.
        earlier = close_lookup.loc[close_lookup.index <= input_ts]
        if earlier.empty:
            print(f"[seed_demo] SKIP {label}: no master data before {input_ts.date()}")
            return
        reference_price = float(earlier.iloc[-1])
    else:
        reference_price = float(close_lookup.loc[input_ts])

    rows = [
        _synthetic_row(
            horizon_days=7, input_ts=input_ts, reference_price=reference_price,
            target_ts=input_ts + pd.Timedelta(days=7),
            pred_return=pred_return_7, prob_up=prob_up_7,
            pred_close=reference_price * (1 + pred_return_7),
        ),
        _synthetic_row(
            horizon_days=30, input_ts=input_ts, reference_price=reference_price,
            target_ts=input_ts + pd.Timedelta(days=30),
            pred_return=pred_return_30, prob_up=prob_up_30,
            pred_close=reference_price * (1 + pred_return_30),
        ),
    ]
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "summary.csv"
        df.to_csv(csv, index=False)
        persist_forecast(
            csv, model_phase=label, db_path=db_path,
            code_source=Path(efp.__file__),
            notes=f"demo seed: {label} ({days_ago}d ago)",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--master-data-csv", type=Path, default=MASTER_CSV)
    parser.add_argument("--wipe", action="store_true",
                        help="Drop existing rows before seeding")
    args = parser.parse_args()

    if args.wipe and args.db.exists():
        print(f"[seed_demo] wiping {args.db}")
        args.db.unlink()

    # Mixed hits/misses so the accuracy badges and history table look realistic.
    #  - 90d:  predicted UP, big miss on h=30 (teaches user that the model fails sometimes)
    #  - 60d:  predicted UP, both correct
    #  - 30d:  predicted DOWN, both wrong
    #  - today: bullish forecast, not yet resolved (hero card)
    _seed_run(args.db, args.master_data_csv, days_ago=90, label="demo_90d",
              pred_return_7=+0.015, pred_return_30=+0.08,
              prob_up_7=0.62, prob_up_30=0.71)
    _seed_run(args.db, args.master_data_csv, days_ago=60, label="demo_60d",
              pred_return_7=+0.025, pred_return_30=+0.055,
              prob_up_7=0.68, prob_up_30=0.64)
    _seed_run(args.db, args.master_data_csv, days_ago=30, label="demo_30d",
              pred_return_7=-0.018, pred_return_30=-0.04,
              prob_up_7=0.32, prob_up_30=0.28)
    _seed_run(args.db, args.master_data_csv, days_ago=0, label="demo_today",
              pred_return_7=+0.012, pred_return_30=+0.038,
              prob_up_7=0.58, prob_up_30=0.62)

    # Backfill everything that has a resolvable target.
    close_lookup = _load_eth_close_lookup(args.master_data_csv)
    conn = connect(args.db)
    try:
        resolved = resolve_pending_forecasts(conn, close_lookup)
        snapshots = refresh_accuracy_snapshots(conn)
        print(f"[seed_demo] resolved={resolved} snapshots={snapshots}")
    finally:
        conn.close()

    print(f"\n[seed_demo] Next step: run export_json, then open public/index.html in a browser")


if __name__ == "__main__":
    main()
