# ETH Forecast

Public research dashboard: https://etherforecast.live/

The active `signal_pipeline` produces 6-hour, 1-day, 3-day, 7-day, 14-day and 30-day ETH forecasts from closed Coinbase ETH-USD and BTC-USD hourly bars. Saved models run hourly; incumbent checkpoints refresh monthly through the weekly/monthly research workflow. Separate experimental candidates require prospective evidence before promotion. Availability does not establish predictive accuracy.

## Operations and storage

- `event_hourly.yml`: incremental collection, immutable issuance, settlement, all-horizon evidence report, snapshot and verified publication.
- `event_watchdog.yml`: checks actual public timestamps and all six windows; bounded recovery when workers are idle. An existing account-level hourly automation provides another trigger.
- `daily_forecast.yml`: daily auxiliary source collection and retired-forecast settlement. Macro/on-chain data is collected separately and is not yet an input to the hourly model.
- `data/event-ledger`: original forecasts, outcomes and verified receipts; `lake/event-inputs` preserves public source receipt vintages and content-addressed active checkpoints.
- `data/daily-forecast`: deployed static website, daily master, availability metadata and normalized market/vendor CSV caches.
- Actions state: two-day hourly snapshots and up to 90-day daily recovery backups; restore chooses the newest trusted hourly or daily copy. Research artifacts still expire after 30 days. Plan dedicated object storage before sustained archive growth.

[Recovery runbook](ops/event_recovery.md) · [Current audit](research/system_audit_20260912/research.md) · [Implementation plan](research/system_audit_20260912/plan.md)

Run `python scripts/audit_event_system.py --root lake/signals` to report source readiness and all six live evidence cohorts without fitting or issuing forecasts.

## Legacy daily collector / model reference

The following documentation describes the reusable daily collector and retired daily model examples. It is not the production hourly inference entrypoint.

## ETH Data Lake + Forecast
ETH forecasting now assumes a data-lake-first workflow:

1. Run `eth_data_collector.py` to build and update the master daily dataset.
2. Train forecasts from `eth_master_daily.csv`.
3. Keep outputs compact by default.

Main files:
- `eth_data_collector.py`: builds the ETH data lake and `eth_master_daily.csv`
- `eth_price_forecast.py`: dual-horizon (`7d` + `30d`) forecast runner
- `eth_forecast_7d.py`: dedicated 7-day runner
- `eth_forecast_30d.py`: dedicated 30-day runner

### Drive Layout
Recommended Google Drive layout:

```text
<drive-root>/
  lake/
    raw/
      market/
        market_data_cache.csv
      manual/
        funding.csv
        onchain.csv
        sentiment.csv
      vendor/
        binance_eth_funding_daily.csv
        deribit_eth_funding_daily.csv
        coingecko_global_daily.csv
        ...
    gold/
      eth_master_daily.csv
      eth_master_schema.csv
      external_feature_summary.csv
    reports/
      collector_source_status.csv
      collector_summary.json
  outputs/
    compact_dashboard.csv
    integrated_report.csv
    summary.json
  prediction_history.csv
```

### Collected Data Groups
`eth_data_collector.py` is designed to keep gathering richer ETH features over time:

- Market OHLCV:
  `ETH`, `BTC`, `SOL`, `BNB`, `SPY`, `QQQ`, `gold`, `oil`, `DXY`, `VIX`, `TNX`
- CoinGecko market structure:
  coin price, market cap, total volume, global crypto market cap, global volume, BTC/ETH dominance, DeFi snapshot
- FRED macro:
  Treasury curve, SOFR/EFFR/IORB, inflation expectations, credit spreads, STL FSI, Fed balance sheet, TGA, RRP, CPI, M2, unemployment
- Sentiment:
  Alternative.me Fear & Greed daily history
- Binance derivatives:
  funding history, recent open-interest history, recent basis history, recent taker buy/sell volume history
- Deribit derivatives/options:
  historical volatility, perpetual funding history, futures snapshot aggregates, options snapshot aggregates, put/call OI ratio
- Etherscan on-chain:
  daily tx count, gas used, network utilization, new addresses, tx fee stats when API access allows it
- Manual external CSVs:
  any numeric daily CSV dropped under `lake/raw/manual/`

Actual collected columns are saved to:
- `lake/gold/eth_master_schema.csv`
- `lake/reports/collector_source_status.csv`

### Install
Required:
`pip install yfinance pandas scikit-learn numpy`

Optional for SHAP:
`pip install shap`

### 1. Build The Master Dataset
Google Drive / Colab example:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
%pip install -q --upgrade yfinance pandas scikit-learn numpy shap
```

```bash
python eth_data_collector.py \
  --period max \
  --interval 1d \
  --drive-root /content/drive/MyDrive/eth_forecast
```

With API keys:

```bash
python eth_data_collector.py \
  --period max \
  --interval 1d \
  --drive-root /content/drive/MyDrive/eth_forecast \
  --coingecko-api-key $COINGECKO_API_KEY \
  --coingecko-pro \
  --etherscan-api-key $ETHERSCAN_API_KEY
```

Optional source toggles:

```bash
python eth_data_collector.py \
  --period max \
  --interval 1d \
  --drive-root /content/drive/MyDrive/eth_forecast \
  --no-binance \
  --no-deribit
```

Manual CSVs can be supplied either by:
- placing them under `/content/drive/MyDrive/eth_forecast/lake/raw/manual/`
- or passing `--manual-features-csv funding.csv onchain.csv sentiment.csv`

### 2. Run Dual-Horizon Forecasts From The Master Dataset
Compact output is now designed to keep only the two main CSV reports plus `summary.json`.

```bash
python eth_price_forecast.py \
  --interval 1d \
  --horizons 7 30 \
  --drive-root /content/drive/MyDrive/eth_forecast \
  --master-data-csv eth_master_daily.csv \
  --compact-output
```

The same command works locally:

```bash
python eth_price_forecast.py \
  --interval 1d \
  --horizons 7 30 \
  --master-data-csv H:/codex/lake/gold/eth_master_daily.csv \
  --output-dir H:/codex/outputs \
  --compact-output
```

### 3. Run A Single Horizon
7-day only:

```bash
python eth_forecast_7d.py \
  --drive-root /content/drive/MyDrive/eth_forecast
```

30-day only:

```bash
python eth_forecast_30d.py \
  --drive-root /content/drive/MyDrive/eth_forecast
```

### Forecast Design Notes
- The dual runner still evaluates both regression and direction classification.
- The direction forecast prefers the best positive event-backtest model over the raw validation leaderboard when possible.
- Probability thresholds are selected from backtests and reused in the final signal.
- Legacy ETH dataset ideas were folded in where they fit the new architecture:
  FRED macro inputs, Fear & Greed sentiment, and VWAP / volume-profile features (`POC`, `VAL`, `VAH`, `Naked POC`).
- Compact mode keeps only:
  `compact_dashboard.csv`, `integrated_report.csv`, and `summary.json`
- Full detailed per-horizon artifacts remain available by omitting `--compact-output` or using `--full-output` in the single-horizon runners.

### Practical Daily-Data Guideline
- hard minimum in code: about 200 cleaned rows
- better for `7d`: about 2 years minimum, 3 years+ preferred
- better for `30d`: about 3 years minimum, 5 years+ preferred

### Notebook Example
```python
from eth_price_forecast import run_suite, summarize_artifacts

artifacts = run_suite(
    interval="1d",
    horizons=[7, 30],
    master_data_csv="/content/drive/MyDrive/eth_forecast/lake/gold/eth_master_daily.csv",
    output_dir="/content/drive/MyDrive/eth_forecast/outputs",
    compact_output=True,
)
summary = summarize_artifacts(artifacts)
summary["horizons"]["7"]
```
