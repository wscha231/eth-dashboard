# Changelog

All notable changes to the ETH price forecasting pipeline and companion
website (`forecast_site/`) are tracked here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates
are UTC.

---

## [Unreleased] · 2026-09-03 (offline lead-signal feature contract)

### Added

- Added checksum-pinned full monthly Binance ETH/BTC spot and perpetual
  backfill for offline review, with complete-UTC-day order-flow, basis,
  intraday-risk, and cross-asset aggregates.
- Added one-day-lagged DefiLlama liquidity features, immutable feature
  manifests, fold-local imputation/standardization, and future-mutation tests.
- Added direct, factorized large-move/direction, multiclass tail, and
  diagnostic barrier-label helpers.
- Added a path-limited GitHub Actions contract that builds and verifies the
  full review artifact without changing the daily forecast path.

### Safety

- Production use, model training, public JSON, database, UI, notifications,
  and the promoted 7-day/30-day models remain unchanged.

## [Unreleased] · 2026-08-27 (compact 30-day regression challenger)

### Added

- Added an opt-in 30-day CatBoost regression challenger that retains the
  leading 192 features from each leakage-safe, train-only fold ranking. The
  incumbent keeps its 360-feature budget in the same run for matched OOF
  comparison.
- Added a focused long-run mode that can select regression estimators and
  disable the classification head, plus a 30-day moving-block bootstrap gate
  against both the incumbent CatBoost model and the no-change anchor.
- Added focused compact evaluation. Pull requests run three recent folds;
  the 36-fold gate requires an explicit `compact_h30_full` manual dispatch
  after intermediate evidence passes. Daily forecasts and scheduled weekly
  evaluation remain unchanged.

### Changed

- Pinned the live 30-day point forecast to the champion promoted by the latest
  authoritative 36-fold OOF gate. The champion remains the no-change anchor
  (RMSE **544.83**, with every learned point model worse), so a noisy short-CV
  daily run can no longer replace it with a weaker regressor. Direction,
  confidence, and uncertainty-range heads continue independently.
- Kept the pinned no-change point forecast separate from uncertainty: its
  lower and upper bounds now come from an independently selected learned
  model's conformal residual interval, with the anchor included in the range.
- Hardened compact-candidate promotion by requiring all expected matched OOF
  rows for both baselines. A compact-only manual dispatch no longer starts the
  unrelated generic evaluator in parallel.
- Repaired daily history merging so incoming observations always fill older
  missing cells even when they fall before the rolling overwrite window. The
  daily workflow now restores the deployed master and seeds its ignored raw
  market cache from that durable copy before refreshing. Cache refresh now
  detects the oldest missing 24/7 ETH date and starts there instead of blindly
  limiting itself to the recent lookback, repairing the observed 2026-04-19
  through 2026-07-11 interval without a repeated full-history download.
- Made the persistence smoke select the newest input date whose 7-day and
  30-day targets are both resolved instead of assuming a fixed 60-day offset.

### Evaluation

- On the three latest deployed-data folds, the compact challenger reduced
  price RMSE from **631.09** to **584.04** versus the incumbent and beat the
  **632.00** no-change anchor. This is smoke evidence only; production remains
  unchanged pending broader purged OOF validation.
- The required 12-fold intermediate gate rejected the compact challenger: it
  improved RMSE only **0.44%** versus incumbent CatBoost but was **16.72%**
  worse than the no-change anchor. The 36-fold compact run was therefore
  skipped and the candidate remains evaluation-only.

## [Unreleased] · 2026-08-26 (LightGBM challenger and stability evidence)

### Added

- Added regularized LightGBM regression and classification challengers, based
  on the strongest transferable result from the reviewed crypto forecasting
  studies. They are opt-in through `ETH_ENABLE_CHALLENGER_MODELS=1` and run in
  model-evaluation workflows only; the daily production forecast remains on
  the promoted registry until a full 36-fold gate approves the challenger.
- Added fold-level feature-selection stability evidence for both the return
  and actionable-direction targets. Checkpoints now retain exact selected
  features by fold plus 50%/80% stability counts and top frequencies, and the
  parallel full-gate merge combines the evidence across all 36 folds.

### Changed

- Extracted estimator construction and horizon-specific parameters from the
  monolithic CLI into `forecasting/model_registry.py`, while preserving the
  existing public `make_models` and `make_classification_models` interfaces.
- Added model-registry and real LightGBM fit/calibration tests to both PR smoke
  evaluation paths.
- Persisted the exact active estimator registry and parameters in long-run
  checkpoints. Resume now discards legacy or mismatched folds instead of
  silently reporting a newly enabled challenger as if it had run all folds;
  parallel merges reject mixed registries as well.
- Resolved optional-model provenance at summary-export time and broadened the
  model-evaluation trigger to `tests/**`, so direct challenger runs are
  reported accurately and registry-test-only changes still execute CI.

### Evaluation

- Latest deployed-data 36-fold gate passed for both 7-day and 30-day horizons
  in Actions run 58. LightGBM did not beat the no-change point anchor and its
  30-day direction head was unstable, so it remains evaluation-only and was
  not promoted into the daily production registry.

## [Unreleased] · 2026-08-24 (7-day direction score normalization)

### Changed

- Replaced the fixed 7-day direction-regime probability overlay with a
  train-only empirical-CDF mapping. Each raw classifier score is expressed as
  its percentile against the final estimator's most recent 180 training
  scores, preventing probability-scale drift between expanding OOF folds.
- Kept the already-passing 30-day calibration and overlay path unchanged after
  a horizon-specific ablation showed the new mapping hurt its recent folds.
- Cached fold-internal feature selection by target and train positions in the
  long-run runner, eliminating repeated correlation/ranking work without
  changing selected features.
- GitHub Actions smoke evaluations now run folds 33-35 (the newest three
  purged OOF windows) instead of folds 0-2, so PR and manual smoke checks
  measure the current market regime rather than the oldest validation slice.
- Corrected chunk progress accounting so folds 33-35 are recorded as three
  completed folds with resume cursor 36, rather than being misreported as a
  complete 36-fold run.

Latest-data local ablation (three most recent 30-row folds, fast model set):

- 7-day Random Forest direction: ROC AUC **0.9111**, balanced accuracy
  **0.8333**, Brier **0.2249**.
- 7-day Extra Trees direction: ROC AUC **0.9000**, balanced accuracy
  **0.8333**, Brier **0.1969**.
- The matching fixed-overlay run peaked at balanced accuracy **0.5444** in the
  same folds.

Full purged OOF (36 × 30-day folds, deployed data through 2026-08-24):

- 7-day Random Forest direction: balanced accuracy **0.5854**, ROC AUC
  **0.6407**, Brier **0.2405**.
- 7-day trimmed classification ensemble: balanced accuracy **0.5863**.
- Selective Random Forest signals at probability 0.55: **59.03%** accuracy,
  **33.09%** actionable-day coverage, **227** signals.
- The full candidate gate passed. Point-price forecasts still do not beat the
  no-change anchor consistently, so the improvement is promoted as direction
  and confidence skill rather than as a reliable point-price edge.

### Follow-up hardening

- Empirical-CDF values are now retained as separate directional ranking
  scores. Threshold selection, signals, and AUC use those scores, while Brier
  scoring and user-facing `probability_up`/confidence use a monotone
  label-frequency calibration shrunk toward the held-out base rate.
- Restored the signed post-candle live-price adjustment for 7-day forecasts as
  a standalone, bounded ±1.5 percentage-point correction without
  reintroducing the rejected momentum/RSI/regime overlay.
- In the 36-fold core-model replay, balanced-accuracy weighting of the
  directional scores reached `0.5980` balanced accuracy and `0.6381` ROC AUC;
  its separately calibrated probability blend recorded `0.2472` Brier. At a
  symmetric `0.60` decision threshold, 92/686 actionable dates emitted a
  signal with `59.78%` accuracy.
- Purged the internal probability-calibration boundary by the full forecast
  horizon, preventing fit labels from reaching into the held-out calibration
  tail for both 7-day and 30-day classifiers.
- Long-run resume and ensemble paths now fall back from `direction_score_up`
  to `probability_up` per row, preserving legacy checkpoint history instead
  of discarding older folds after a resumed run. Recovered scores are written
  back into the checkpoint rows before SQLite/public export.
- Backtest schema v4 persists and exports `direction_score_up`, so published
  OOF rows can reproduce the exact threshold decisions shown by the site.
- Live forecast summaries, SQLite history, and public JSON now retain
  `classification_direction_score_up` together with its signal threshold.
- Direction confidence is capped by the calibrated probability of the class
  actually selected by the direction score, avoiding opposite-class
  confidence when the ranking score and event probability straddle 0.5.

---

## [Unreleased] · 2026-04-22 (parallel tracks: overlay kill + LLM analyst)

### Added — `ETH_OVERLAY_DISABLE_*` env switches on regression OOF

Walk-forward OOF diagnostic (`tmp_analyze_h30.py`) traced the negative
skill of `trimmed_regression_ensemble_equal` to a cascade of shrinkage
layers rather than a single model/feature problem:

- h=7 skill vs naive: **-10.6%** (RMSE $282 vs naive $255)
- h=30 skill vs naive: **-12.6%** (RMSE $612 vs naive $543)
- std(predicted_return) / std(actual_return) = **26.6% at h=30**
- corr(pred, reference_close) = 0.983 at h=30 — model essentially echoes
  current price

Per-model leaderboard (`tmp_leaderboard_h30.py`) confirmed swapping the
chart model alone recovers only 2.3% RMSE; catboost ($598) vs ensemble
($612) — still worse than naive. The cause is structural:
`apply_regime_response_overlay` blends up to 55% toward momentum
anchors, then `skill_weighted_trimmed_mean` drops max+min models per
row, then MAE loss shrinks toward the conditional median.

`eth_price_forecast.py` now exposes a kill switch for the overlay on
the two diagnostic paths without touching production:

```
ETH_OVERLAY_DISABLE=1                 # disables in every context
ETH_OVERLAY_DISABLE_WALK_FORWARD=1    # only walk_forward_leaderboard
ETH_OVERLAY_DISABLE_HOLDOUT=1         # only build_recent_holdout_report
```

Default (no env): overlay active, zero behaviour change. The live /
hybrid / headline forecast paths intentionally ignore the flag — their
production semantics require the anchor stabilisation. The first
overlay-off longrun OOF run is queued behind the in-flight
`phase5_nodefi` job; result will inform whether the overlay is a net
positive even for daily headlines.

### Added — `llm_analyst/` experiment: structured market-environment view

Scaffolded a parallel product track that doesn't try to beat the
regression's point-forecast miscalibration by adding more numeric
features — instead asks an LLM to synthesise a structured directional
view from a quantitative snapshot + recent news headlines, and demands
a strict JSON schema (direction, confidence, thesis, key_risks,
would_change_view_if, expected_return_low/high).

Rationale: the data audit showed the numeric pipeline is comprehensive
(~1000 candidate features across FRED macro / BTC cross-asset /
Deribit-Binance derivatives / limited on-chain / Fear&Greed) but the
qualitative regime layer — news, regulation, ETF flows, narrative
shifts — is structurally absent. These are exactly the drivers that
explain the 2026-01→02 crash the regression missed ($3,300 → $1,800).

Modules (decoupled from `eth_price_forecast` and `forecast_site`):

| Module            | Purpose |
| ----------------- | ------- |
| `news_store.py`   | SQLite schema for headlines, UNIQUE on url_sha1, indexed by `published_at` so the backtest harness can date-gate in O(log n). |
| `news_fetcher.py` | RSS pull from CoinTelegraph / Decrypt / The Block / CoinDesk. stdlib XML parser, no `feedparser` dep. Handles RSS 2.0 + Atom. |
| `snapshot.py`     | Date-gated numeric summary of `lake/gold/eth_master_daily.csv` — price, vol, derivatives, macro, on-chain (what's populated), sentiment. Renders to Markdown for prompts. |
| `analyst.py`      | Prompt assembly + Anthropic Messages API via plain HTTPS (no SDK dep). Strict JSON parse, schema validation, and auto-persist to `analyst_views.db`. |
| `backtest.py`     | Walk-forward harness: weekly calls over a date range, each leak-safe. Hard `--max-calls` guardrail. Dry-run mode for prompt iteration at zero API cost. |

Verified end-to-end:

- News fetcher: 133 real headlines ingested from 4 feeds (oldest
  `2025-12-29`, newest `2026-04-22`).
- Snapshot: 2026-04-21 renders to ~1.2KB Markdown with price, vol,
  macro, derivatives, Fear&Greed.
- Prompt total: ~3.8KB / ~1k tokens — roughly $0.05 per view on Claude
  Sonnet 4.5.
- Backtest dry-run correctly samples the 2026-01→03 window with
  actuals ranging -38.7% to +10.3%, including the crash that broke the
  regression model.

Not yet live: historical news backfill (RSS gives only ~4 months),
site panel integration, daily cron hook. Next step is setting
`ANTHROPIC_API_KEY` and running a capped POC (≤20 calls ≈ $1) to
produce the first skill-vs-naive number for the LLM path.

### Diagnostic scripts (not deployed)

- `tmp_analyze_h30.py` — loads `backtest_longrun_history.json`, prints
  RMSE/MAE/MAPE, naive-baseline skill, predicted-return aggressiveness,
  regime bucket errors, top-10 worst misses.
- `tmp_leaderboard_h30.py` — per-model RMSE ranking from
  `backtest.json`, highlights where `chart_model` sits.

Both gitignored under `tmp_*.py`; included for reproducibility of the
conclusions above without committing.

---

## [Unreleased] · 2026-04-22 (deploy branch auto-sync with main)

### Fixed — index.html changes on main never reached Vercel

Structural CI bug: `.github/workflows/daily_forecast.yml` committed
JSON updates to `data/daily-forecast` (the branch Vercel deploys
from) but never merged main's code changes onto it. Result: the
deploy branch's `index.html` was frozen at 2026-04-20 (587 lines)
while main had 1584 lines of new frontend work. Everything added
this session — methodology card, regime grid, version timeline,
A/B diff, worst-cases, mobile responsive, the 3-yr main-chart
history merge — was invisible to users visiting etherforecast.live.

**Immediate rescue** (commit `67a2a84` + `55c51c8` on
`data/daily-forecast`): manually merged main → data/daily-forecast
and force-added the backtest JSONs
(`backtest.json`, `backtest_versions.json`,
 `backtest_longrun_history.json`,
 `backtest_predictions_*.json`) that are gitignored on main.

**Permanent fix** (this commit, on main):
`daily_forecast.yml` now runs `git merge origin/main --no-edit`
after checking out the data branch, before staging the new JSON
blobs. A 3-way merge works cleanly because main's `.gitignore`
keeps the tracked-on-data-only JSONs out of main's history entirely.
If the merge ever conflicts (shouldn't, but defensively) the step
logs a warning, aborts the merge, and continues with a data-only
update so the cron doesn't red-light.

---

## [Unreleased] · 2026-04-22 (main-chart 3-year history fix)

### Fixed — Predicted-vs-actual cards were showing only ~2 months

The two "Predicted vs actual · h=N" cards on `public/index.html` were
wired to `history.json` (daily cron resolved forecasts), which only
has data since the cron came online in Feb 2026. Users visiting the
site saw a sparse chart starting Feb 2026 and asked where the
historical backtest predictions were. The 3-year OOF data was
already present in `backtest_longrun_history.json` but only fed the
separate "3-year backtest · leakage-safe" cards.

`renderChart()` now accepts an optional `oofBundle` and merges it
with the live rows:

- OOF walk-forward points (1080 × 2 horizons, 2023-05 → yesterday)
  are the primary source — they're the leakage-safe reconstruction
  the Methodology card describes.
- Live cron resolved forecasts top up the tail beyond the OOF end.
- Dedup by target date; OOF wins on overlap.
- Dense timelines (>200 points) render as thin lines without per-point
  dots so 1000+ points don't collapse into a visual blob.

Subtitles updated to reflect the new scope
("3-year walk-forward OOF backtest + live resolved daily runs").

---

## [Unreleased] · 2026-04-22 (backtest archive + track-record site)

Two-goal release: (1) give a skeptical viewer an auditable "this is
what we actually achieved" track record, (2) let A/B phase comparisons
feed back into model improvement.

### Added — Backtest archive DB layer (schema v2 + v3)

- **`forecast_site/schema.sql`** gains three new tables:
  - `backtest_runs`       — one row per ingested freeze (model_phase,
                            frozen_utc, mode, freeze_json_sha1).
  - `backtest_metrics`    — long-form leaderboard rows
                            (per phase × horizon × head × model).
  - `backtest_predictions` — per-date OOF predictions (longrun only;
                             ~2000 rows per phase for the 36 × 30 grid).
  UNIQUE (`model_phase`, `freeze_json_sha1`) drives idempotent ingest;
  `ON DELETE CASCADE` lets the longrun-refresh path (DELETE+INSERT)
  atomically replace a phase's rows without orphaning children.
  `schema_version` now records `(1, 2, 3)` so future migrations can
  detect which step a DB is at.

### Added — Backtest ingest + export pipeline

- **`forecast_site/persist_backtest.py`** — reads
  `tests/phase0/*_metrics.json` freezes into the DB.
  - Idempotent: a re-run of the same (phase, sha1) is a no-op.
  - Longrun detection fires on EITHER `mode.startswith("longrun_")`
    OR filename `*_longrun_oof_metrics.json` — the two signals are
    independent on purpose (no hard coupling between filename and mode).
  - Longrun refresh uses `BEGIN IMMEDIATE` + DELETE+INSERT so concurrent
    readers (the webserver) never see a half-updated row set.
  - NaN floats in leaderboards (ensemble `component_models` etc.)
    coerce to SQL NULL — the SQL aggregates the frontend reads would
    otherwise collapse to the string `"nan"`.
- **`forecast_site/export_backtest_json.py`** — DB → JSON, split so the
  frontend only downloads what's visible:
  - `backtest.json`                  — headline (best reg/cls per
                                        phase × horizon) + flat matrix.
  - `backtest_versions.json`         — run metadata for the version
                                        timeline chart.
  - `backtest_longrun_history.json`  — ensemble-only predicted-vs-actual
                                        points for the 3-yr chart +
                                        regime grid.
  - `backtest_predictions_<phase>.json` (lazy-loaded) — full per-phase
                                        OOF rows for the A/B diff card
                                        and worst-case drill-down.
  Empty-DB safe: every export returns an empty container, not an error,
  so the frontend's fallback "not yet available" states are reached.
- **`forecast_site/persist_and_export_longrun.py`** — thin driver that
  stitches `persist_backtest` + `export_backtest_json` for local reruns
  after a phase finishes (the CI workflow does the same two steps).

### Added — Frontend backtest UI (`public/index.html`)

Single-file, vanilla-JS, Chart.js via CDN. All cards render empty-state
fallbacks when the DB is cold, so the page never hard-errors on a fresh
deploy.

- **Methodology card** — one-paragraph explanation of walk-forward
  + purged + embargoed CV. Sits directly above the metrics so a
  skeptical viewer reads the contract before the numbers.
- **Regime breakdown card** — bucket the 36-fold longrun OOF points
  into bull / bear / chop by realised actual_return, show bucket-level
  directional accuracy + mean absolute return error. Driven purely
  client-side from `backtest_longrun_history.json`; no server changes
  to add regimes.
- **Version timeline** — Chart.js overlay of RMSE (regression head) +
  Brier (classification head) per phase, read from
  `backtest_versions.json` + `backtest.json` headline. A hover surfaces
  the full freeze metadata (mode, cv_test_size, master_data_csv).
- **3-yr ensemble chart + horizon selector** — predicted-vs-actual
  dashed line with horizon (7 / 30) toggle. Wired directly to
  `backtest_longrun_history.json`.
- **A/B diff card** — auto-activates when two phases are present; shows
  ensemble-only RMSE / MAE / directional-accuracy deltas plus a
  Wilcoxon signed-rank p-value pill (styled `.significant` when p<0.05).
  Short-circuits gracefully when we only have one phase or the two
  picked phases match.
- **Worst-case anecdotes panel** — top-k miss predictions (|pred-actual|
  / actual) per horizon, rendered as a grid of cards. Surfaces the
  2026-02 drawdown as the single worst h=30 miss (~96% over-estimate),
  which doubles as honest disclosure of the model's bull bias.
- **Mobile responsive** — `@media (max-width: 600px)` block collapses
  all multi-column grids to `minmax(0, 1fr)`, shrinks the hero price
  font, and wraps all tables in `.table-wrap` for horizontal scroll
  instead of layout-breaking overflow.

### Added — CI wiring

`.github/workflows/daily_forecast.yml`:

- **`Persist backtest archive`** step — runs `persist_backtest.py`
  against the full `tests/phase0/*_metrics.json` glob. Idempotent, so
  the cron safely ingests both existing and newly-committed freeze
  JSONs on every tick.
- **`Export backtest JSON`** now runs `export_backtest_json` after
  `export_json`, producing the four new frontend blobs.
- **`Commit updated DB`** force-adds the new `backtest*.json` family
  (including the large per-phase `backtest_predictions_*.json`) on the
  `data/daily-forecast` branch so the Vercel deploy picks them up.

### Added — Pytest coverage (22 tests)

- **`tests/test_db_schema.py`** (5 tests) — all expected tables
  created, `schema_version == 3`, reconnect is idempotent, FK CASCADE
  works, (phase, sha1) UNIQUE raises `IntegrityError`.
- **`tests/test_persist_backtest.py`** (9 tests) — basic ingest,
  re-ingest is a no-op, different sha creates a new run, longrun
  DELETE+INSERT replaces metrics + predictions, longrun detected by
  mode OR filename, NaN → NULL coercion, smoke test against the real
  baseline fixture.
- **`tests/test_export_backtest_json.py`** (8 tests) — every export
  function's shape contract, "pick best model" logic, longrun-only
  filter, empty-DB safety, and `main()` writes all expected files
  when invoked with `--output-dir`.
- **`tests/conftest.py`** — shared fixtures: `temp_db_path`, `temp_db`
  (connection, yields + closes), and a `tiny_freeze_json` factory that
  produces configurable freeze JSONs (longrun yes/no, predictions,
  n_models, frozen_at) without coupling to the real freeze pipeline.

### Added — Phase 6 onchain depth (feature wiring, no model change yet)

- **`eth_data_collector.py`** — pulls DefiLlama (free, no key:
  stablecoin supply, chain TVL, DEX volume) and Glassnode (optional,
  paid tier) endpoints through the standard vendor-transform pipeline.
  Both emit `__asof / __age_days / _diff_7 / _pct_30 / _zscore_30`
  columns that feed directly into the existing feature builder — no
  special-casing. Graceful skip on 401/404 so a missing Glassnode key
  doesn't red-light the cron.
- **`eth_price_forecast.py`** — adds `defillama_` and `glassnode_` to
  `GENERIC_VENDOR_PREFIXES`. No named regime composite added yet: that
  is deliberately gated behind a LOO ablation (same bar Phase 5 used
  to re-add the macro features).

### Added — Phase 5 / Phase 6 freeze + longrun scripts

- **`tests/phase0/freeze_phase6_production.py`** — Phase 6 production
  freeze with the onchain block unlocked. Serialises `phase6_production
  _metrics.json` for the archive pipeline.
- **`tests/phase0/longrun_oof_common.py`** — shared 36-fold × 30-day
  OOF harness (walk-forward with purge + embargo) used by the three
  longrun drivers below. Each fold emits per-date predictions the
  frontend's 3-yr chart + worst-case panel read.
- **`tests/phase0/longrun_oof_phase5_production.py`**,
  **`…_phase5_nodefi.py`**, **`…_phase6_production.py`** — the three
  heads of the A/B decision: stock phase5 (macro re-added),
  phase5 with DeFi/onchain removed, and phase6 with onchain unlocked.
- **`tests/phase0/diff_longrun_oof.py`** — pairwise Wilcoxon on two
  longrun OOF metric files; prints deltas and a p-value. Drives the
  Go/No-Go decision for each phase re-add.

### Changed

- `forecast_site/schema.sql` v2/v3 DDL runs under the same
  `CREATE TABLE IF NOT EXISTS` / `INSERT OR IGNORE` contract as v1, so
  existing production DBs auto-migrate on the next `connect()`. No
  manual migration needed.
- `.gitignore` now excludes the auto-regenerated
  `forecast_site/public/backtest*.json` family (the DB + phase0 JSONs
  are the sources of truth). CI force-adds them on the deploy branch.

---

## [Unreleased] · 2026-04-20 (staleness feedback)

### Added — Forecast staleness banner

Surfaces the two pieces of information a skeptical user needs before trusting
a displayed number:

1. **Forecast age** — "generated N hours ago" computed in the browser from
   `run.run_timestamp_utc` so it updates without touching the backend. Uses
   minutes / hours / days granularity.
2. **Live vs reference divergence** — `(LIVE_PRICE − reference_price) /
   reference_price`, rendered with an up/down arrow and signed percentage.

A colour-coded banner combines both signals:

| severity | trigger                                  | meaning shown                               |
|----------|------------------------------------------|---------------------------------------------|
| 🟢 fresh | age ≤ 12h AND |divergence| ≤ 3%          | prediction context still valid              |
| 🟡 warn  | age ≤ 36h OR  |divergence| ≤ 8%          | directional signal OK, absolute targets may need re-run |
| 🔴 stale | age >  36h OR |divergence| >  8%         | treat as outdated; wait for next daily run  |

The banner re-renders on every 60-second live-price tick so a drift crosses
the thresholds in real time.

---

## [Unreleased] · 2026-04-20 (afternoon updates)

### Added — Live price + frontend polish

- **Live ETH/USD price** on `public/index.html`. Pulls from Coinbase (primary)
  with CoinGecko fallback, refreshes every 60 seconds, shows a pulsing green
  dot + "N seconds ago" freshness indicator. Both endpoints are free and
  CORS-friendly — no API key needed for any real user.
- **"% from live" on every forecast card**. The forecast is a ~$2,449 target
  7 days out, so showing "+1.20%" against the frozen reference price from
  when the model ran becomes misleading the moment spot moves. The card now
  shows both: `(ref +1.20%)` for the original forecast context and
  `from live $2,284 → +7.21%` for what the prediction is saying to someone
  looking right now. The live figure re-computes on every 60s tick.
- **Predicted-vs-actual charts** per horizon (h=7 and h=30) using Chart.js
  via CDN. No build step, no bundle — the whole page stays a single HTML
  file. Dark theme, tooltips, dashed-line for actuals.

### Added — Alerts

- **`forecast_site/notify_telegram.py`** — Markdown-formatted daily summary
  to Telegram via the Bot API. Reads `TELEGRAM_BOT_TOKEN` and
  `TELEGRAM_CHAT_ID` from env. Gracefully no-ops if either is unset and
  swallows transient Telegram failures — never red-lights the cron.
- Workflow wires the notify step after backfill + export, with
  `continue-on-error: true` for defence in depth.

### Added — Deployment scaffolding

- **`forecast_site/vercel.json`** — zero-build Vercel config. Serves
  `public/` with 60s edge cache on JSON + 5min HTML cache. Instructions in
  the new `forecast_site/README.md` explain how to point Vercel at the
  `data/daily-forecast` branch so the site auto-redeploys on every cron
  tick.
- **`forecast_site/README.md`** — architecture diagram, local dev / seed /
  preview commands, and the Vercel deploy walkthrough.

### Changed

- `run_daily.py` now runs `export_json` + `notify_telegram` as final steps
  (with `--skip-export` / `--skip-notify` for ergonomic local reruns).
- `daily_forecast.yml` force-adds `public/*.json` when committing to
  `data/daily-forecast` (they're gitignored locally but Vercel needs them
  on the deploy branch).

---

## [Unreleased] · 2026-04-20 (morning)

### Added — Version control + changelog

- Initial git repo with `.gitignore` (excludes Python caches, `.bak` files,
  secrets, pickled metrics, and ephemeral JSON/DB artifacts).
- This `CHANGELOG.md` to track pipeline + website evolution together.

### Added — Website infrastructure (`forecast_site/`)

First end-to-end cut of a daily-updating ETH forecast website. The pipeline
now persists every run into a SQLite database, backfills actuals once target
dates pass, and exports compact JSON blobs the frontend consumes directly.

- **`forecast_site/schema.sql`** — five-table schema:
  `forecast_runs` (1) → `forecasts` (N) → `actuals` (1:1),
  plus `accuracy_snapshot` and `schema_version`.
  Idempotent via `UNIQUE (input_timestamp_utc, model_phase)` on runs and
  `UNIQUE (run_id, horizon_days)` on forecasts.
- **`forecast_site/db.py`** — connection helper. Enables `foreign_keys=ON`,
  `journal_mode=WAL`, row factory set, and runs `schema.sql` on every connect
  (safe because all DDL uses `CREATE IF NOT EXISTS`).
- **`forecast_site/persist_forecast.py`** — CSV → DB. Reads
  `eth_forecast_outputs/latest_forecast_summary.csv`, inserts one
  `forecast_runs` row and N `forecasts` rows. Computes `code_version` as a
  sha256 prefix of `eth_price_forecast.py` for auditability.
- **`forecast_site/backfill_actuals.py`** — resolves pending forecasts against
  the master CSV once the target date has a close. Computes direction
  correctness (FLAT predictions treated as abstention, not a miss) and Brier
  contributions, then refreshes rolling accuracy snapshots for window
  sizes `[30, 90, 180, all]` per horizon.
- **`forecast_site/export_json.py`** — writes
  `public/{latest,accuracy,history}.json` from the DB so the frontend never
  runs SQL.
- **`forecast_site/run_daily.py`** — local version of the CI pipeline
  (collector → forecast → persist → backfill). Each stage has `--skip-*` so
  you can rerun just the DB side after a data-collector bug fix.
- **`forecast_site/smoke_test.py`** — E2E synthetic test. Seeds a run 60
  days in the past, asserts persist → backfill resolves both horizons and
  produces Brier values in the expected range. Passing at commit time.
- **`forecast_site/seed_demo.py`** — populates `predictions.db` with four
  synthetic runs (90d / 60d / 30d / today ago) so the static frontend has
  realistic hero, badges, and history even before the first real cron tick.
- **`forecast_site/public/index.html`** — dark-themed static mockup. Hero
  card with bear/base/bull scenarios, rolling accuracy badges
  (🟢 / 🟡 / 🔴), and a 50-row resolved history table. Fetches the three
  JSON blobs and renders with vanilla JS — no build step required.

### Added — CI/CD

- **`.github/workflows/daily_forecast.yml`** — daily cron at `10 0 * * *`
  (00:10 UTC = 09:10 KST, off the top-of-hour rush hitting free data APIs).
  Runs the full pipeline on `ubuntu-latest`, commits the updated
  `predictions.db` + master CSV to a dedicated `data/daily-forecast` branch
  (keeps `main` clean of binary churn), and uploads per-run artifacts for
  14 days of debugging.
- `workflow_dispatch` inputs for `fast_mode` and `model_phase` so you can
  trigger degraded-accuracy reruns manually without editing the file.
- `concurrency: daily-forecast` with `cancel-in-progress: false` — a manual
  trigger during a scheduled run queues behind it instead of killing it.

### Added — Algorithm work (in progress)

- **`tests/phase0/freeze_phase5_loo.py`** — Leave-One-In ablation over the
  six Phase 3B macro composites
  (`fred_yield_curve_change_20`, `fred_real_yield_10y`,
   `fred_real_yield_10y_z_90`, `fred_credit_stress`,
   `fred_credit_stress_change_20`, `macro_asymmetry_20`).
  Recomputes each composite inline from base columns, then layers exactly
  one on top of the Phase 4B baseline. Smoke run (baseline-only) reproduced
  Phase 4B metrics exactly: h=30 brier=0.1728, AUC=0.7505.
- **`tests/phase0/analyze_phase5_loo.py`** — classifies each variant as
  SAFE / WATCH / CULPRIT vs baseline across h=7 and h=30. Tolerances:
  ±0.005 brier, ±0.02 AUC, ±4% RMSE. Strikes-and-severe logic
  (1 severe or 2 strikes ⇒ CULPRIT; 1 strike ⇒ WATCH; else SAFE). Worst
  horizon verdict per feature.
- **`tests/phase0/freeze_phase5_production.py`** — template with empty
  `PHASE5_ALLOWLIST`. Populated after the full LOO pass completes.

### Context — Why Phase 5 exists

Phase 4B restored the h=30 classification baseline (brier 0.173, AUC 0.750)
by removing the six Phase 3B macro composites, which had regressed h=30
brier to 0.266 when all six were active. Phase 5's job is not to re-add the
whole block — it's to identify which subset, if any, was actually causing
the regression, and to re-add only the safe features individually.

---

## [Phase 4B] · 2026-04-18

### Changed

- Dropped all six Phase 3B macro composites from `build_features` in
  `eth_price_forecast.py`. Restores h=30 classification brier from 0.266 to
  0.173 and AUC from 0.699 to 0.750.
- Added Phase 3B removal guards in `tests/phase0/test_leakage_audit.py` to
  prevent accidental re-introduction without LOO evidence.

---

## [Phase 3B] · 2026-04-14 · ⚠ regressed, now removed

### Added (later reverted)

- Six FRED-derived macro composites: yield-curve delta, real 10y yield plus
  its 90-day z-score, credit stress (HY+BBB OAS average) plus its delta,
  and a dollar-vs-rates asymmetry signal. All of these rolled out together,
  masking which one(s) drove the h=30 regression. This is the reason Phase 5
  is running a LOO instead of a bulk re-add.

---

## [Phase 2.5] · earlier

### Added

- Hybrid regression + classification stack, walk-forward CV with embargo,
  signal tiering (STRONG / MODERATE / WEAK / NO_SIGNAL), bear/base/bull
  scenario pricing.

---

<!-- Template for future entries:

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

-->
