# Search discovery research — 2026-09-12

Baseline: main ba038ed5c925109f34a1b03a2d2b17d6095720e1.

## Observed production and repository facts
- The public homepage responds with 200, title `ETH Forecast`, and a description. Canonical, Open Graph, Twitter cards and language navigation are absent.
- `/robots.txt` and `/sitemap.xml` returned 404 during the preceding live audit. Missing robots.txt does not itself forbid crawling.
- The application is plain HTML + JavaScript + Chart.js. `events.js` reads `signals.json`; its six forecast windows are 6/24/72/168/336/720 hours. The primary numerical evidence arrives after JavaScript executes.
- Donation addresses already come from `donations.json`. They must remain exactly as configured; no new wallet or payment recipient will be invented.
- Vercel serves `forecast_site/public` from `data/daily-forecast`, not main. `scripts/publish_events.sh` copies the homepage and event assets on each hourly publication. Other archive publishers also copy the homepage but retain existing public files.
- `scripts/verify_event_site.py` compares production index.html with the main source byte-for-byte. Injecting live values into index.html would break this important verification. A linked, separate static outlook page is safer.
- All state writers share the `daily-forecast` Actions concurrency group. A lightweight asset publishing workflow must use that same group and a current production branch snapshot to avoid overwriting forecasts.
- No AGENTS.md was present in the baseline tree. Open PRs concern models/data and do not overlap this scope.

## Design conclusions
1. Keep the dashboard index deterministic and add readable, independent /outlook.html and /ko/ pages. The outlook is a dated publication snapshot of actual issued records, never an additional prediction.
2. Generate static values only from the production release JSON; validate the shape, timestamps, probability simplex and ordered quantiles. Explain the timestamp and stale/missing state explicitly. Display failed or incomplete horizons as unavailable.
3. Add metadata, robots, sitemap and a self-hosted preview image; do not add tracking or third-party payment code.
4. Improve support navigation and address copy controls, preserving network names and addresses.
5. Owner verification tokens for Google/Naver/Bing require the user's own accounts. No account ownership token, registration, account creation, social post or ranking result can be fabricated.
6. A site-only workflow can ship SEO assets without collecting data or retraining. The hourly publisher must regenerate the static snapshot from each new release.

## Sources
- https://developers.google.com/search/docs/crawling-indexing/javascript/javascript-seo-basics
- https://support.google.com/webmasters/answer/9008080?hl=ko
- https://searchadvisor.naver.com/guide/request-feed
- https://www.bing.com/webmasters/help/add-and-verify-site-12184f8b

## Concurrent update reconciled
While this work was in progress, main fe43884193f2b74536d5a9ee37606bde07c24fd8 added sitemap/robots publication and verification plus a private draft privacy notice. This change retains the new publisher checks, tests, README and non-public privacy draft. The sitemap now extends the homepage entry with the Korean guide and dated outlook. No incomplete privacy notice is published.
