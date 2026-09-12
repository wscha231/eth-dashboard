# forecast_site

## Top animation and service status

The top panel plays the original 5-second, muted, looping MP4 only when `events.js` validates all six current forecast
windows and `site_status.json` confirms `auto` mode. An operating light means
current forecasts are available; it does not certify accuracy or active computation.
Maintenance, unknown status, connection failure and delayed data pause the video
and display a dim static poster. The actual MP4 is an owned deployment asset,
not a link to a video-generation service. The panel is 240px wide on desktop
and 192px on small screens. A pause/play button respects manual pausing and
provides recovery when a browser blocks automatic playback.
Reduced motion and a hidden tab pause the effects without changing service status.

To enter maintenance, edit `forecast_site/public/site_status.json` on main:

```json
{"schema_version":1,"mode":"maintenance","message":"Forecast service maintenance"}
```

To resume automatic status use `{"schema_version":1,"mode":"auto","message":""}`.
The file is a static deployment asset: a repository edit takes effect only after
a successful deployment. The page polls it every 60 seconds independently of the
large replay download. A failed status request turns the light off and preserves
any last confirmed maintenance instruction. An invalid or stale forecast cannot
be made operational by `auto` alone. Reloading starts with lights off.

The search, hourly-event and completed-hybrid publication paths all use the
asset list in `scripts/publish_search_assets.py`, including the MP4 and status file from
main. A status-only edit triggers the search-assets workflow. Respect
`ops/deployment_cooldown.json`; do not enable deployment early. After publication,
check the public status JSON, the visible badge, actual motion and maintenance
transition before claiming the live site has changed.

The donation rationale sits immediately above the existing wallets. API figures
are dated candidate budgets, not confirmed expenses. The 3% MAE / 5% Brier figures
are proposed pilot hurdles, not measured or expected gains. Review official prices,
endpoint coverage and public-use licences before a purchase. See
`docs/brand-status-animation/research.md` for sources and limitations.

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

## Search discovery

The public sitemap is `https://etherforecast.live/sitemap.xml`. It lists
canonical public pages, not data files or dashboard anchor links. Add new
pages only when they are publicly available. No synthetic `lastmod` date is
emitted; hourly data refreshes must not invent page modification timestamps.

`robots.txt` allows crawling and advertises the sitemap. The hourly publisher
copies both files to the deployment branch and verifies their served contents.
In Google Search Console, select the `etherforecast.live` property, open
**Sitemaps**, and submit the full sitemap URL (or just `sitemap.xml` if the
form already displays the site's URL prefix). Google controls crawling,
indexing, and ranking; publishing or submitting the sitemap does not confirm
any of those outcomes.

The published privacy policy is `public/privacy.html`, served at
`https://etherforecast.live/privacy.html`. It is linked in the homepage, Korean
guide, dated outlook and sitemap. The search-assets publisher and hourly
publisher retain it on the deployment branch. `verify_site_privacy.py` checks
the live canonical URL, meaningful HTML body and homepage link after publication;
hourly exact-release verification also compares its bytes with the source.

The policy identifies the public ETH Forecast maintainer (`wscha231`) and uses
the existing GitHub support route. Public requests must contain only a request
for private contact. For Google authorizations, the consent screen's support
email is an additional private contact route. No contact email was invented.
The old `drafts/privacy.html` remains an unpublished historical draft.

Google Drive backup/restore is not yet implemented in the hourly workflow.
The policy explicitly labels this operator-only integration as preparation and
describes the handling required before enabling it: least-privilege access,
private tokens and working files, retention/deletion, permitted sharing and
Google Limited Use. Implementing or changing the connection requires updating
the disclosure to match actual scopes, recipients and retention. Do not use
Google Workspace APIs to train generalized or non-personalized AI/ML models.
Do not describe publication of these pages as Google OAuth verification approval.

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
