# EtherForecast DATA update policy

Updated: 2026-09-16

This policy makes DATA collection continuous rather than a one-off backfill. The machine-readable source list is `config/data_source_registry.json`.

| Layer | Cadence | Purpose | Durable path |
|---|---:|---|---|
| ETH/BTC operational market/event data | hourly | 6h/1d live input and issuance ledger | event-hourly + Drive |
| Bitget/Hyperliquid current OI/context | every 4h | prospective short-horizon leverage/context history | `data/daily-forecast` + Drive after run |
| Free historical derivatives / DVOL / macro research | twice daily | basis, funding, DVOL, Hyperliquid history, FRED/ALFRED | `data/daily-forecast` + Drive after run |
| Daily market/DeFi/stablecoin/vendor master | existing twice-daily daily job | gold/master refresh and settlements | daily-data + Drive |
| Freshness guard | every 6h | detect missing/stale active sources against registry SLA | `lake/reports/data_refresh_status.*` + Drive sweep |
| Drive archive sweep | every 6h and after source workflows | immutable content-addressed source/forecast archive | EtherForecast-Storage |
| Weekly maintenance | Sunday review from freshness/catalog reports | gap review, source failures, planned-source priority | GitHub/Drive reports |
| Monthly maintenance | monthly deep review | history coverage, PIT/revision semantics, rights, restore integrity, MODEL feedback | research checkpoint |

## Freshness rules

- Fast derivative context: warning after 8h, hard stale after 16h.
- Daily crypto/DeFi/market sources: generally warning after 36-48h and hard stale after 72-96h.
- Macro PIT data: wider SLA because underlying series are weekly/monthly, while the collector itself still polls twice daily.
- Planned sources are visible in the registry but do not fail collection jobs until an actual collector is activated.
- A fresh source is not automatically MODEL-eligible. Coverage/unit/availability checks and matched-origin ablation remain mandatory.

## Current planned-but-not-yet-complete sources

- historical liquidations
- additional historical OI
- PIT-safe DFF, RRP and 10Y-2Y histories
- historical option skew / IV term structure / put-call OI
- Ethereum issuance/burn, staking/validator, blob/fee and exchange-flow history
- ETF/institutional flows with release timestamps and rights metadata

When one of these is acquired, add its collector, hot-cache file, cadence and SLA to the registry; do not create a separate ad-hoc storage convention.
