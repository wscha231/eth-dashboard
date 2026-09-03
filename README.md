# R1000 Top30 Institutional (Colab)

## Files
- `r1000_top30_institutional.py`: Full pipeline implementation.
- `R1000_Top30_Institutional_Colab.ipynb`: One-click Colab runner.
- `eth_price_forecast.py`: ETH-USD daily price forecast example.

## Quick Start (Colab)
1. Create `/content/drive/MyDrive/r1000_top30_institutional`.
2. Upload both files above into that folder.
3. Open `R1000_Top30_Institutional_Colab.ipynb`.
4. Update `cfg['sec_user_agent']` to your real contact email.
5. Run all cells.

If you see `ModuleNotFoundError: pandas_market_calendars`, run:
`%pip -q install pandas-market-calendars` and rerun imports.

If yfinance transiently fails and many tickers get blacklisted, delete:
`<base_dir>/cache_misc/yf_fail_tickers.json` and rerun.

If you previously ran an older build, regenerate `feature_store` after updating code.
Older builds clipped large numeric fields too aggressively and may drop all rows in training.

## Public Interfaces
- `run_all(cfg) -> dict`
- `build_universe_monthly(cfg) -> DataFrame`
- `build_feature_store(cfg) -> DataFrame`
- `train_walkforward(cfg, features) -> ModelBundle`
- `backtest_portfolio(cfg, signals) -> BacktestResult`
- `export_outputs(cfg, artifacts) -> dict[str]`

## Outputs
Saved under `<base_dir>/outputs/`:
- `top30_latest.csv`
- `scored_latest.csv`
- `top30_explain_latest.csv`
- `weights_latest.json`
- `backtest_metrics.json`
- `equity_curve.csv`
- `run_summary.json`

Phase-0 audit reports are saved under `<base_dir>/outputs/reports/`.
Acceptance test summary is saved as `<base_dir>/outputs/reports/acceptance_checks.json`.

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

### Offline Lead-Signal Source Preflight

The lead-signal research path is isolated from `eth_data_collector.py` and the
daily forecast. It verifies official Binance monthly 1h archives, backfills
compact DefiLlama daily source tables, and records OKX/Deribit feasibility:

```bash
python scripts/backfill_lead_signals.py \
  --as-of-date 2026-08-30 \
  --max-binance-archives 12
```

Outputs:

- `lake/manifests/lead_signal_sources.json`: immutable URLs, upstream/local
  SHA-256 values, source coverage, and bounded archive validation;
- `lake/reports/lead_signal_source_readiness.json`: strict-JSON offline gate;
- `lake/gold/lead_signal_sources/`: compact full-history DefiLlama source
  tables that are not read by the promoted model.

Raw Binance ZIP samples remain under ignored `lake/raw/lead_signal/`. The
manifest is intentionally not overwritten unless `--replace-outputs` is
passed. A passing preflight approves only offline feature work; it does not
approve production use, model promotion, or public output changes.

### Offline Lead-Signal Daily Features

After the source preflight passes, build the PR2 review artifact from all four
checksum-pinned monthly streams through their latest common complete month:

```bash
python scripts/build_lead_signal_features.py --workers 6
```

The builder creates:

- `lake/gold/lead_signal_daily.csv.gz`: compact daily order-flow, basis,
  intraday-risk, BTC-leadership, and Ethereum-liquidity features;
- `lake/manifests/lead_signal_features.json`: immutable archive checksums,
  UTC availability rules, feature groups, and output hash;
- `lake/reports/lead_signal_feature_readiness.json`: the strict offline gate
  for the next source-ablation evaluation.

Every Binance day requires all 24 hourly bars to close before the declared UTC
cutoff. DefiLlama values are delayed by one feature day because historical
publication vintages are unavailable. Missing-value imputation, eligibility,
and standardization are fitted separately inside each outer training fold.
The accompanying target helpers add direct upside/downside tails, a factorized
large-move/direction target, and a three-class tail label. Barrier labels are
diagnostic only.

This command does not modify or feed the promoted daily forecast. Raw monthly
ZIPs remain under ignored `lake/raw/lead_signal_full/`; production use remains
blocked until later matched-date Gate A/B evaluation and terms review pass.

The 2026-09-03 locked PR2 build validated 374 archives and produced 3,269
daily rows with 177 columns through 2026-07-31. Its four-stream common window
contains 2,388 eligible days, with the first authoritative PR3 test date set
to 2021-12-31. Historical hourly-grid anomalies are declared in the manifest
and their 34 affected UTC days are quarantined. Floating outputs use ten
significant digits so identical pinned inputs rebuild to identical bytes.

### Offline Lead-Signal Source Ablation

PR3 compares every source-augmented candidate with the same model family on
the same dates. Gate A is a six-block engineering smoke; Gate B is a manual,
expanding calendar-year OOF evaluation with a three-day purge, prior-only
calibration, and prior-only alert thresholds:

```bash
python scripts/evaluate_lead_signal_ablation.py \
  --profile smoke \
  --data-git-ref origin/data/daily-forecast \
  --nonlinear histgradient \
  --output /tmp/lead-signal-gate-a.json
```

The frozen Gate B run covered 1,672 OOF dates in 2022-2026. The best all-lead
HistGradientBoosting candidate improved AP from 0.03204 to 0.05964 and Brier
score from 0.03437 to 0.02883 against its matched core-only baseline, but it
failed the absolute promotion contract: episode recall was 34.78%, false
alerts were 3.39 per 90 days, and 66.84% of aggregate Brier gain came from one
calendar block. The candidate therefore remains offline. Full compact
evidence is in `tests/phase0/lead_signal_source_ablation_metrics.json`.

Pull-request CI keeps all six matched 30-day folds but bounds the automatic
smoke to the direct logistic core/all-leads pair. This exercises the data,
fold-safety, calibration, threshold, and serialization contracts within the
10-minute infrastructure budget. The predeclared nonlinear registry is only
rerun through manual `full` dispatch; its frozen Gate A/B evidence remains the
authoritative performance record.

This evaluator does not write the model registry, daily outputs, database,
site files, or notifications. A failed Gate B is evidence to retain the data
layer and stop model promotion, not permission to relax thresholds.

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
