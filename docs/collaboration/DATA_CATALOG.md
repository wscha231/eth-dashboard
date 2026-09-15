# EtherForecast DATA catalog

Updated: 2026-09-16

This catalog tracks **what has actually been collected**, not what a vendor claims may be available. Historical backfills are reconstructed research unless original receipt/vintage evidence exists. Rows and ranges below come from verified collection reports; they are not evidence of predictive edge.

| Family | Source / feature set | Actual stored range | Rows | Frequency / width | Timing status | Research role |
|---|---|---:|---:|---|---|---|
| Core market | ETH/BTC long hourly research input | 2017-01-01 → 2026-09-14 cutoff | ETH 85,001 / BTC 84,986 | hourly | reconstructed historical source; missing-hour audit available | baseline price/volume, trend, realized volatility, relative strength |
| DeFi | DefiLlama Ethereum TVL | 2017-09-27 → 2026-09-15 | 3,276 | daily, 1 col | reconstructed backfill | network capital/DeFi activity candidate |
| DeFi | DefiLlama Ethereum DEX volume | 2018-11-02 → 2026-09-15 | 2,875 | daily, 1 col | reconstructed backfill | on-chain trading/activity candidate |
| Stablecoins | DefiLlama Ethereum stablecoin history | 2017-11-29 → 2026-09-15 | 3,213 | daily, 2 cols | reconstructed backfill | liquidity/stablecoin impulse candidate |
| Derivatives | Bitget ETH mark/index basis | 2019-08-10 → 2026-09-14 | 2,593 | daily, 5 cols | reconstructed backfill | futures dislocation/basis candidate |
| Derivatives | Bitget ETH funding | 2026-06-18 → 2026-09-15 | 90 | daily aggregates, 3 cols | public endpoint returned short history | leverage/carry candidate; insufficient alone for long folds |
| Derivatives | Bitget ETH current context/OI | 2026-09-15 → accumulating | 1 at first verified run | daily snapshot | prospective receipt only | OI/current leverage state; never synthesize old history |
| Options / volatility | Deribit ETH DVOL | 2021-03-24 → 2026-09-15 | 2,002 | daily, 7 cols | reconstructed official-index backfill | forward volatility/range/uncertainty candidate |
| Derivatives | Hyperliquid ETH funding/premium | 2023-05-12 → 2026-09-15 | 1,223 | daily, 5 historical cols | reconstructed public history | funding/premium/leverage candidate |
| Derivatives | Hyperliquid ETH current context/OI | 2026-09-15 → accumulating | 1 at first verified run | daily snapshot, 7 cols | prospective receipt only | native OI, funding, premium, mark/oracle/mid, notional volume |
| Macro | FRED current-vintage requested set | through 2026-09-15 | 12,641 | long-form, 10 series | **revised hindsight history**, not PIT-safe by itself | research/reference only until release-lag treatment |
| Macro PIT | FRED/ALFRED initial-release reconstruction | 2017-01-01 → 2026-09-15 as-of grid | 6,369 long / 3,545 as-of days | 7 successful series | date-only release usable next UTC day by conservative policy | PIT macro/liquidity candidate |
| Macro PIT gaps | DFF / RRPONTSYD / T10Y2Y initial-release route | unresolved | — | 3 series | HTTP 400; do not substitute revised history silently | open DATA request |
| Options snapshots | Existing Deribit future/option snapshots | short, accumulating | short | daily snapshots | prospective/short reconstructed caches | skew/PCR/OI research after adequate history |

## Storage checkpoints

- Authoritative historical DATA research release: `eth-data-18525a7b61ee33af20467989`, SHA256 `921d8e150296ce1c9fc52ab699eb8b171870efe80ad2845f4674a2f057e7c7ef`.
- Current hot research branch: `data/daily-forecast`; verified free-source data commit `22677303293edab8d832760a87f6af63af3c6a59`.
- Latest expanded free-source Drive checkpoint: `DATA-free-source-expanded-20260916-b3a94c0b3f2f.zip`, Drive ID `12QLsTS14EuImEuKLCS_K2WYWpKX-oZ_x`, 398,591 bytes, SHA256 `b3a94c0b3f2fd47ccf8265283a18d8904e1b9643a6b98671f8823f547dfc8e83`.
- Previous free-source checkpoint remains immutable: `DATA-free-source-fallback-state-20260916-5fc865a248fe.zip`, Drive ID `1tdQ2LumbicpO9cVE7oIoPyrh7wXnzW7C`.

## Eligibility rules

1. A collected column is not automatically a model feature.
2. Historical backfills without original receipts remain reconstructed research.
3. Current revised macro history is not treated as historical point-in-time information.
4. OI/options snapshots with no free history accumulate prospectively; missing old values are not imputed as observations.
5. Exchange/contract/unit identity must remain attached to derivatives data. Bitget, Hyperliquid, Binance and Deribit values are not interchangeable just because labels look similar.
6. A feature family enters MODEL experiments only after coverage, unit and availability checks. It enters production only after matched-origin horizon-specific ablation beats the predeclared baseline/gate.
7. Public/commercial redistribution requires a separate rights review; API accessibility alone is insufficient.

## Next collection priorities

1. Historical liquidation data from a legal/free source; otherwise accumulate live liquidation events prospectively.
2. Additional historical OI sources to complement prospective Bitget/Hyperliquid snapshots.
3. PIT-safe DFF, RRP and 10Y-2Y release history.
4. ETH option skew / IV term structure / put-call OI history beyond DVOL.
5. ETH issuance/burn, staking/validator state, fee/blob activity and exchange flows with reproducible timing.
6. ETF/institutional flows and stablecoin issuance/flow histories with release timestamps and rights metadata.
