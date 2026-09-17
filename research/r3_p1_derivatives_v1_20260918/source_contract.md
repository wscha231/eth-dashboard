# R3 P1 derivatives source contract — 2026-09-18

## Purpose

This contract freezes the current derivative/options research boundary before any new P1 model experiment. It does **not** authorize production MODEL use, website use, public redistribution, or paid-product use.

The repository already contains overlapping collectors. Therefore the default is to audit and reuse only what is eligible; do not add a duplicate collector merely because a source is available.

## Existing collection paths

### Bitget

- Long/reconstructed research path: `free_source_backfill.yml` -> `bitget_eth_free_features.csv`.
- Fast prospective path: `fast_derivative_snapshots.yml` -> `bitget_eth_context_4h.csv`.
- Current fast collector records receipt-timestamped public snapshots and explicitly does not synthesize historical OI.
- The 2026-09-17 15:00 UTC run had 7 accumulated fast rows, beginning 2026-09-16 00:00 UTC.

### Hyperliquid

- Long/reconstructed funding path: `free_source_backfill.yml` -> `hyperliquid_eth_free_features.csv`.
- Fast prospective context: `fast_derivative_snapshots.yml` -> `hyperliquid_eth_context_4h.csv`.
- Funding history is historical provider data. OI/mark/oracle/premium snapshots are prospective receipt-time observations and must not be backfilled synthetically.
- The 2026-09-17 15:00 UTC run had 7 accumulated fast rows, beginning 2026-09-16 00:00 UTC.

### Deribit

- Current reconstructed DVOL path: `free_source_backfill.yml` -> `deribit_eth_dvol_daily.csv`.
- Legacy daily paths also collect short reconstructed funding plus prospective futures/options snapshots.
- PR #26 demonstrated that the funding endpoint can expose much longer history when queried in bounded windows: 62,254 / 62,256 hourly intervals from 2019-08 through 2026-09, with two hours on 2020-08-27 still missing. This is useful provenance research, but it is not a production data license or a prospective record.
- PR #26 remains a research reference only and must not be merged merely to obtain longer history.

## PIT / history rules

1. Provider historical responses are reconstructed history. Their event timestamps are not EtherForecast receipt times and must never be labeled as prospective performance.
2. Fast snapshots use actual collector receipt time as `available_at`.
3. Historical OI and liquidation amounts must never be synthesized from prices, candles, volume, funding, or other proxies.
4. A successful HTTP response is not a complete backfill. Requested interval continuity must be checked explicitly.
5. Daily DVOL becomes eligible only after its source candle has closed; it is implied-volatility history, not a reconstructed options surface.
6. Historical funding, basis and DVOL cannot enter an R4 backtest until their publication/availability semantics and source rights have passed a separate gate.

## Rights boundary

Rights were rechecked on 2026-09-18 against current provider terms.

### Deribit

Current Deribit Exchange Membership Terms / applicable client terms state that market data and derived data are for personal use and that aggregation, resale, publication, forwarding or other processing beyond personal use requires prior written approval. Therefore:

- `rights_status = nonpersonal_market_or_derived_use_requires_prior_written_approval`
- public redistribution: blocked
- MODEL use: blocked
- website/API use: blocked
- paid-product use: blocked
- historical alpha use: blocked pending written approval plus PIT audit

Reference: https://support.deribit.com/hc/en-us/articles/25944532191645-Deribit-Exchange-Membership-Terms-Deribit-FZE

### Bitget

Current Bitget API terms grant a limited API license and restrict unauthorized derivatives, repackaging/resale and other exploitation of API-related data; current general terms also restrict commercial use unless agreed in writing. Therefore:

- `rights_status = commercial_or_derived_use_requires_written_authorization_or_agreement`
- public redistribution: blocked pending explicit permission review
- MODEL/site/paid-product use: blocked
- existing public-branch persistence must not be treated as authorization

References:
- https://www.bitget.com/support/articles/12560603797947
- https://www.bitget.com/support/articles/360014944032/

### Hyperliquid

No sufficiently specific current official market-data reuse grant was verified in this audit. Public endpoint availability alone is not a commercial redistribution license. Therefore:

- `rights_status = rights_unreviewed_public_api`
- public redistribution: blocked
- MODEL/site/paid-product use: blocked
- prospective collection may continue only as research while rights and storage exposure are reviewed

## Public-storage exposure

The current research workflows persist some vendor-derived CSVs on the public `data/daily-forecast` branch. That historical implementation is **not** evidence that redistribution is permitted.

This contract freezes the rule that no new P1 provider is allowed to copy this pattern until rights are explicitly cleared. A separate storage migration must move restricted/unreviewed provider research to the private immutable Drive path before the corresponding source can be considered for longer prospective accumulation.

The governance registry records these sources as fail-closed while that migration/rights review is pending.

## Frozen promotion gates

A P1 derivative source may leave research-only status only if all applicable gates pass:

1. source contract and exact field/unit semantics documented;
2. rights status explicitly permits the intended non-personal/commercial use;
3. raw and transformed data storage does not violate redistribution restrictions;
4. PIT `available_at` policy is auditable;
5. historical coverage/continuity and revision behavior are measured;
6. feature hypothesis is preregistered before the experiment;
7. same-origin ablation beats the frozen baseline under the R4 metric gate;
8. prospective evidence is accumulated before production promotion.

Failure of any gate leaves the source blocked. No post-hoc feature rescue or model promotion is allowed.
