# DATA checkpoint — 2026-09-16

## Core DATA release and merged research infrastructure

DATA PR #54 and MODEL PR #53 are merged to `main`. The original authoritative DATA research release remains `eth-data-18525a7b61ee33af20467989` with release SHA256 `921d8e150296ce1c9fc52ab699eb8b171870efe80ad2845f4674a2f057e7c7ef`. It is a pinned historical research release, not a live-market promotion.

Original DATA verification remains valid: ETH 85,001 / BTC 84,986 hourly bars on an 85,039-hour common grid; 32 incumbent features reproduced exactly; 18 optional hourly research extensions; all six horizons reproduced archived source/target checks and metrics. The release remains `partial`, `reconstructed`, and rights-unreviewed for public redistribution. Predictive edge is not implied by data readiness.

## Actual free-source collection completed

Merged PR #55 added the first free historical backfill path. Main workflow run `34958191975` completed successfully. It established:

- DefiLlama Ethereum chain TVL: 3,276 daily rows, 2017-09-27 through 2026-09-15.
- DefiLlama Ethereum DEX volume: 2,875 daily rows, 2018-11-02 through 2026-09-15.
- DefiLlama Ethereum stablecoin history: 3,213 daily rows / 2 columns, 2017-11-29 through 2026-09-15.
- Direct Bybit backfill: HTTP error on the GitHub runner; do not treat Bybit as available there.
- First monolithic FRED initial-release request: HTTP error; replaced by per-series collection below.

Merged PR #58 added resilient free-source fallbacks. Main workflow run `34990686756` completed successfully.

Merged PR #59 (`c88905f30d7a0deb058d025e066232cfef4ba853`) added Deribit ETH DVOL and Hyperliquid ETH funding/current-market context. Main workflow run `35035984882` completed successfully and pushed hot DATA commit `22677303293edab8d832760a87f6af63af3c6a59` to `data/daily-forecast`.

Latest actual collection at 2026-09-15 UTC:

### Derivatives and volatility

- Bitget mark/index basis history: 2,593 daily rows / 5 columns, 2019-08-10 through 2026-09-14. This is a Bitget mark-versus-index basis proxy, not exchange-wide basis.
- Bitget funding history: 90 daily aggregates / 3 columns, 2026-06-18 through 2026-09-15. Do not infer that older funding is unavailable everywhere; this is what the tested public endpoint returned through this collector.
- Bitget OI/current context: one current daily snapshot is accumulated prospectively. Historical OI is **not synthesized**.
- Combined Bitget research frame: 2,594 daily rows / 17 columns, 2019-08-10 through 2026-09-15.
- Deribit ETH DVOL: 2,002 daily rows / 7 columns, 2021-03-24 through 2026-09-15. Native DVOL OHLC is retained; derived research columns are 1-day change, 7-day mean and 30-day z-score.
- Hyperliquid ETH funding/premium history: 1,223 daily rows / 5 historical columns, 2023-05-12 through 2026-09-15.
- Hyperliquid current context: one current snapshot with native OI, funding, premium, mark/oracle/mid price and daily notional volume. OI/current-context history accumulates prospectively and is **never backfilled synthetically**.
- Combined Hyperliquid research frame: 1,223 daily rows / 16 columns, 2023-05-12 through 2026-09-15.

### Macro and liquidity

- FRED current-vintage history: 12,641 long-form observations across all ten requested series with no series-level errors. This is latest revised history and is **not** safe historical PIT evidence by itself.
- FRED/ALFRED initial-release history: 6,369 long-form observations and a 3,545-day as-of state from 2017-01-01 through 2026-09-15 for the successfully reconstructed series. Date-only releases become eligible the next UTC day by policy.
- Initial-release collection remains partial: `DFF`, `RRPONTSYD`, and `T10Y2Y` returned HTTP 400 on this route and are not silently replaced with revised historical values.
- Successful initial-release/as-of series are CPI, unemployment, M2, Fed balance sheet, TGA, SOFR and HY OAS. SOFR starts later than the 2017 study start; HY OAS PIT coverage in the constructed table is also shorter.

## Durable storage

`data/daily-forecast` is the hot research cache. Latest data commit: `22677303293edab8d832760a87f6af63af3c6a59`.

The latest expanded free-source artifact contains seven files and has SHA256 `b3a94c0b3f2fd47ccf8265283a18d8904e1b9643a6b98671f8823f547dfc8e83`.

It is stored in the existing private Drive root as:

- File: `DATA-free-source-expanded-20260916-b3a94c0b3f2f.zip`
- Drive file ID: `12QLsTS14EuImEuKLCS_K2WYWpKX-oZ_x`
- Size: 398,591 bytes
- Contents: Bitget features, Deribit DVOL, Hyperliquid features, FRED current-vintage long data, FRED initial-release long data, FRED PIT/as-of daily state, collection report.
- Verification: the GitHub artifact, Drive object, and Drive re-download have the same size and full SHA256.

An earlier five-file fallback checkpoint remains stored as `DATA-free-source-fallback-state-20260916-5fc865a248fe.zip` (Drive ID `1tdQ2LumbicpO9cVE7oIoPyrh7wXnzW7C`, SHA256 `5fc865a248fe202418c8e4d69bf15fd74a7600f25948ea175a0db101d4254ea6`). It is retained as an immutable earlier checkpoint rather than overwritten.

Historical backfills remain reconstructed research unless an original historical receipt/vintage is available. Public API accessibility does not by itself establish commercial redistribution rights.

## Current readiness for MODEL research

Available/useful now as research candidates:

1. Long ETH/BTC price and volume history.
2. Long DefiLlama Ethereum TVL, DEX volume and stablecoin history.
3. Bitget mark/index basis back to 2019-08-10.
4. Deribit ETH DVOL back to 2021-03-24.
5. Hyperliquid ETH funding/premium history back to 2023-05-12.
6. Partial but causally safer FRED/ALFRED initial-release macro state for seven series.
7. Short Bitget funding plus prospectively accumulating Bitget and Hyperliquid OI/current-context snapshots.
8. Existing Deribit funding/options/futures snapshots which continue accumulating prospectively.

These data are **not automatically promoted into the forecasting model**. MODEL must run matched-origin, horizon-specific ablations and compare against the same baselines before a feature family can be considered useful.

## Still open

- Acquire or reconstruct PIT-safe `DFF`, `RRPONTSYD`, and `T10Y2Y` release histories without substituting revised hindsight values.
- Find additional free historical OI and liquidation sources; keep exchange, contract and unit identities separate.
- Historical liquidation backfill remains weak: do not infer liquidation history from price/volume or fabricate it.
- Expand historical option-chain/skew/term-structure data beyond DVOL if a legal/free source is found; otherwise accumulate official snapshots prospectively.
- Add Ethereum staking, issuance/burn, fee/blob, validator and exchange-flow data from sources with adequate history and acceptable rights.
- Review source licensing before any commercial/public redistribution.
- Integrate only validated columns into the gold feature set after coverage/unit/availability checks and MODEL ablation evidence.

MODEL should treat all additions as research candidates. No source is promoted solely because it has longer history or more columns.
