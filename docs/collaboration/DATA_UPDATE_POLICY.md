# EtherForecast DATA update policy

Updated: 2026-09-16

This policy makes DATA collection continuous rather than a one-off backfill. The machine-readable source inventory is `config/data_source_registry.json` plus `config/data_source_registry_existing_supplement.json` for legacy caches already present before the registry was introduced.

| Layer | Cadence | Purpose | Durable path |
|---|---:|---|---|
| ETH/BTC operational market/event data | hourly | 6h/1d live input and issuance ledger | event-hourly + Drive |
| Bitget/Hyperliquid current OI/context | every 4h | prospective short-horizon leverage/context history | `data/daily-forecast` + Drive after run |
| Long derivatives / DVOL / macro research | twice daily incremental | refresh recent overlap only; preserve historical cache and rolling-feature warmup | `data/daily-forecast` + Drive after run |
| Daily market/DeFi/stablecoin/vendor master | existing twice-daily daily job | gold/master refresh and settlements | daily-data + Drive |
| Freshness guard | every 6h | detect missing/stale active sources and unregistered raw CSVs | `lake/reports/data_refresh_status.*` + Drive |
| Full historical reconciliation | every Sunday | re-fetch long free history to repair gaps/provider corrections and audit registry coverage | `data/daily-forecast` + Drive |
| Monthly reconciliation checkpoint | first day of month | full history + freshness/PIT/coverage checkpoint and durable restore evidence | GitHub artifact + Drive |
| Drive archive sweep | every 6h and after source workflows | immutable content-addressed source/forecast archive | EtherForecast-Storage |

## Incremental-refresh rules

- Initial collection is full when a source cache is absent.
- Normal twice-daily long-source refresh requests recent history only. Derivative/DVOL collectors use a 60-day overlap; the first 30 days are warmup so 30-day rolling features can be recomputed before new values overwrite recent cache rows.
- FRED current-vintage and initial-release collection uses a 180-day recent window during ordinary refresh. The complete accumulated long tables remain on disk.
- Hyperliquid pagination is paced during multi-page history requests. Weekly/monthly reconciliation can still request the full history.
- New non-null feature values may replace cached values, but a partial provider failure or NaN cannot erase an already stored non-null value for the same date/column.
- Provider failure never deletes the historical cache. The collection report records the error and keeps the prior evidence.
- Historical OI/liquidation values are never synthesized solely to fill a time range.

## Freshness rules

- Fast derivative context: warning after 8h, hard stale after 16h.
- Daily crypto/DeFi/market sources: generally warning after 36-48h and hard stale after 72-96h.
- Macro PIT data: wider SLA because underlying series are weekly/monthly, while the collector itself still polls twice daily.
- Planned sources are visible in the registry but do not fail collection jobs until an actual collector is activated.
- Any CSV discovered under `lake/raw/market` or `lake/raw/vendor` but missing from the combined registry is reported in `unregistered_files`.
- A fresh source is not automatically MODEL-eligible. Coverage/unit/availability checks and matched-origin horizon-specific ablation remain mandatory.

## Concurrent persistence rules

DATA collectors may finish at nearly the same time. They must not copy or replace the entire hot-data directory. Each writer uses `scripts/persist_data_branch_overlay.py` to:

1. fetch the latest `data/daily-forecast` tip;
2. overlay only that collector's generated files;
3. preserve unrelated source files;
4. retry from the newest branch tip if a concurrent push wins first;
5. treat identical repeated output as a successful no-op.

This behavior was added after the first continuous live run exposed non-fast-forward conflicts between fast, long-refresh and freshness writers.

## Drive durability

`data/daily-forecast` is the hot source branch. The existing `daily-data` Drive stream is the durable source archive. Source workflows trigger the Drive archive workflow after completion, and the independent six-hour archive sweep remains a second line of defense. Existing content-addressed Drive snapshots are not overwritten or manually pruned.

## Current planned-but-not-yet-complete sources

- historical liquidations
- additional historical OI
- PIT-safe DFF, RRP and 10Y-2Y histories
- historical option skew / IV term structure / put-call OI
- Ethereum issuance/burn, staking/validator, blob/fee and exchange-flow history
- ETF/institutional flows with release timestamps and rights metadata

When one of these is acquired, add its collector, hot-cache file, cadence, timing status and freshness SLA to the registry; do not create a separate ad-hoc storage convention.
