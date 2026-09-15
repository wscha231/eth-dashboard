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

Merged PR #58 added resilient free-source fallbacks. Main workflow run `34990686756` completed successfully and pushed data commit `e275d960c3605dd6795746fb27eb6185a82a41e7` to `data/daily-forecast`.

Actual fallback collection at 2026-09-15 UTC:

- Bitget mark/index basis history: 2,593 daily rows, 2019-08-10 through 2026-09-14. This is a free historical mark-versus-index basis proxy, not exchange-wide basis.
- Bitget funding history: 91 daily aggregates, 2026-06-17 through 2026-09-15. Do not infer that older funding is unavailable everywhere; this is what the tested public endpoint returned through this collector.
- Bitget OI: one daily ETH-unit snapshot on 2026-09-15. Historical OI is **not synthesized**; it accumulates prospectively from scheduled runs.
- Combined Bitget research frame: 2,594 daily rows / 17 columns, 2019-08-10 through 2026-09-15.
- FRED current-vintage history: 12,636 long-form observations across all ten requested series with no series-level errors. This is latest revised history and is **not** safe historical PIT evidence by itself.
- FRED/ALFRED initial-release history: 6,369 long-form observations and a 3,545-day as-of state from 2017-01-01 through 2026-09-15 for the successfully reconstructed series. Date-only releases become eligible the next UTC day by policy.
- Initial-release collection remains partial: `DFF`, `RRPONTSYD`, and `T10Y2Y` returned HTTP 400 on this route and are not silently replaced with revised historical values.

The successful initial-release as-of series are CPI, unemployment, M2, Fed balance sheet, TGA, SOFR, and HY OAS. HY OAS has PIT coverage in the constructed 2017+ table only from 2023-09-20; SOFR begins 2019-04-02; the other successful series have a carried PIT state from the 2017 research start after prior releases.

## Durable storage

The new free-source fallback artifact contains five files and has SHA256 `5fc865a248fe202418c8e4d69bf15fd74a7600f25948ea175a0db101d4254ea6`. It is stored in the existing private Drive root as `DATA-free-source-fallback-state-20260916-5fc865a248fe.zip`, Drive file ID `1tdQ2LumbicpO9cVE7oIoPyrh7wXnzW7C`. The GitHub artifact and Drive re-download are both 280,090 bytes and their full SHA256 hashes match.

`data/daily-forecast` is the hot research cache; Drive remains the durable copy. Historical backfills remain reconstructed research unless an original historical receipt/vintage is available.

## Current readiness and next DATA work

Available/useful now for feature research:

1. Long ETH/BTC price and volume history.
2. Long DefiLlama TVL, DEX volume and stablecoin history.
3. Bitget mark/index basis back to 2019-08-10.
4. Partial but causally safer FRED/ALFRED initial-release macro state for seven series.
5. Short Bitget funding history and a prospectively accumulating OI snapshot series.
6. Existing short Deribit funding/IV/options/futures snapshot caches.

Still open:

- Acquire or reconstruct PIT-safe `DFF`, `RRPONTSYD`, and `T10Y2Y` release histories without substituting revised hindsight values.
- Find additional free historical OI/funding/liquidation sources; keep exchange/contract/unit identities separate.
- Integrate validated Bitget research columns into the gold feature set only after coverage/unit/availability review.
- Continue prospectively accumulating OI/options snapshots in Drive.
- Add release-lag metadata and feature-family ablation evidence before MODEL promotion.
- Review source licensing for any future commercial/public use.

MODEL should treat these additions as research candidates and run matched-origin ablations by horizon. No new source is promoted solely because it has longer history.
