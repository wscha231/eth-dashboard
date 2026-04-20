# Changelog

All notable changes to the ETH price forecasting pipeline and companion
website (`forecast_site/`) are tracked here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); dates
are UTC.

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
