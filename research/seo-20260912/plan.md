# Search and support implementation plan — 2026-09-12

## Authorization and scope
The user quoted and accepted the preceding audit and requested GitHub setup plus concrete platform registration instructions. This implements those scoped recommendations. Model changes, social account creation/posting, paid services, payment recipients and unprovided verification tokens are outside this change.

## Files and behavior
- `forecast_site/public/index.html`: descriptive title, canonical/Open Graph/Twitter metadata, links to Korean guide and dated static outlook, support navigation, copy-address controls.
- `forecast_site/public/ko/index.html`, `site-info.css`: Korean service/forecast interpretation/support guide, with explicit link to the English dashboard.
- `forecast_site/public/robots.txt`, `sitemap.xml`, `social-preview.svg`, `social-preview.png`: crawl discovery and a self-hosted, non-promissory share card. Sitemap includes only canonical content pages.
- `scripts/publish_search_assets.py`: copy an explicit allowlist to a target checkout; render outlook.html from that checkout's signals.json; optional git staging of those paths only. Missing data produces an honest unavailable page. Input record prices/probabilities are never recalculated or saved back.
- `scripts/publish_events.sh`: invoke the publisher after copying new signals, so every hourly release has matching static content.
- `.github/workflows/site_search.yml`: small main-push/manual asset workflow, using the existing daily-forecast concurrency group and preserving all production data. No paid APIs or inference.
- `.github/workflows/event_hourly.yml`: include the snapshot builder in path triggers.
- `tests/test_search_publication.py`: meaningful cases for exact source values, escaped text, malformed/incomplete data, stale state, idempotence and retaining data/verification files.
- `research/seo-20260912/registration.md`: account-specific steps, remaining prerequisites and result status.

## Trade-offs
A separate pre-rendered snapshot avoids modifying the verified dashboard HTML at publication time. Korean content is a guide, not a claim that the entire interface has been translated. Explicit asset copying preserves unknown verification files already on production. The preview image is branded text/artwork, never a fabricated performance chart.

## Verification and release
- Test the publisher against fixtures and the real current production payload; confirm the production source data is unchanged.
- Check JavaScript syntax, shell syntax, metadata, sitemap XML and preview at desktop/mobile widths if the runtime supports it.
- Open a scoped PR, run the required repository CI, then merge and verify production URLs, actual metadata and snapshot provenance. Report registration as pending until the user's accounts verify ownership and accept submissions.
