# forecast_site

The public-facing half of the ETH price-forecasting project. The ML pipeline
(`../eth_price_forecast.py`) runs once a day in GitHub Actions; this package
turns every run into durable database rows and exports JSON blobs that a
static frontend can render — no server needed.

```
    eth_price_forecast.py                (daily, GitHub Actions)
            │
            ▼
    latest_forecast_summary.csv
            │
            ▼  persist_forecast.py
    ┌──────────────────────┐
    │   predictions.db     │  SQLite, WAL, FK on
    │  ─ forecast_runs     │
    │  ─ forecasts         │  1:N
    │  ─ actuals           │  1:1 with forecasts
    │  ─ accuracy_snapshot │  rolling 30 / 90 / 180 / all windows
    └──────────────────────┘
            │         │
            │         └──── backfill_actuals.py  (resolve + score once target passes)
            ▼
    export_json.py
            │
            ▼
    public/{latest,accuracy,history,health}.json
            │
            ▼
    public/index.html   (vanilla JS + Chart.js, no build step)
```

The homepage keeps two evidence streams separate:

* `history.json` is the daily, post-deployment track record. A chart point is
  added only after the forecast target has resolved.
* `backtest_longrun_candidate_history.json` is the newest weekly OOF
  candidate, published on PASS or FAIL.
* `model_eval_latest.json` is the newest candidate gate.  The production
  forecast continues to use `model_eval_last_pass.json`; a failed candidate
  is visible but cannot silently become the production gate.
* `health.json` reports live forecast, resolved-history, OOF-evaluation and
  data-quality freshness independently.

## Running the full pipeline locally

```bash
# One command, matches what CI runs.
python -m forecast_site.run_daily
```

Individual stages (each can be skipped):

```bash
python -m forecast_site.run_daily \
    --skip-collector \
    --skip-forecast \
    --master-data-csv lake/gold/eth_master_daily.csv
```

## Previewing the frontend

```bash
cd forecast_site/public
python -m http.server 8765
# open http://127.0.0.1:8765/
```

`fetch()` calls for the JSON blobs will fail from `file://` due to CORS —
you need a real HTTP server even for local preview.

## Demo data for the frontend

Before the first real cron tick, seed synthetic runs so the page isn't
empty:

```bash
python -m forecast_site.seed_demo --wipe
python -m forecast_site.export_json
```

This gives you a hero card, accuracy badges for h=7 / h=30, resolved history
table, and populated predicted-vs-actual charts.

## Deploying to Vercel

The static bundle lives in `public/`. A `vercel.json` at the `forecast_site/`
root tells Vercel:

1. No build command — the HTML is already built.
2. Serve `public/` as the root.
3. Short cache TTLs on the JSON (60s edge, 10min stale-while-revalidate) so
   the frontend always picks up fresh cron outputs within a minute.

Deploy flow (first time, one-time):

```bash
# Option A — Vercel CLI
npm i -g vercel
cd forecast_site
vercel                         # link to Vercel project, answer prompts
vercel --prod                  # first production deploy

# Option B — web UI
# 1. Create a new Vercel project
# 2. Connect the GitHub repo
# 3. Set "Root Directory" = forecast_site
# 4. Set "Production Branch" = data/daily-forecast
#    (so Vercel redeploys every time the cron commits fresh JSON)
```

After that, every GitHub Actions run that bumps `public/*.json` on the
`data/daily-forecast` branch triggers an automatic Vercel redeploy — no
further action needed.

The daily job verifies the deployed `health.json` after pushing. A separate
13:30 KST watchdog dispatches at most one recovery run when the site is stale
and no daily run is already active.

## Schema notes

* Idempotent per `(input_timestamp_utc, model_phase)` — safe to rerun the
  daily pipeline multiple times without duplicate runs.
* FLAT predictions are treated as **abstentions**
  (`direction_correct = NULL`) — the model isn't penalised for refusing to
  call a flat tape. This mirrors how `hybrid_signal_tier = NO_SIGNAL`
  behaves in the forecast code.
* `accuracy_snapshot.window_days = 9999` means "all-time". The frontend
  surfaces it as `all`.

## CI integration

See `../.github/workflows/daily_forecast.yml`. The workflow commits the
updated `predictions.db` + exported JSON blobs back to the
`data/daily-forecast` branch — that branch is what Vercel should track.
