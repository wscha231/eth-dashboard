# Approved repair scope — 2026-09-14

The current instruction explicitly requests all improvements and production publication, authorizing implementation and deployment of this bounded repair.

1. forecast_site/public/index.html: visible bilingual privacy link directly below header, with concise public-service/data-purpose description; retain footer link and approved animation.
2. forecast_site/public/privacy.html: clarify EtherForecast / ETH Forecast identity; update date; add plain-language overview of information, purpose, handling/retention and deletion. Preserve accurate planned-versus-running disclosure and established contact routes.
3. scripts/verify_site_privacy.py: permit validation of a specified static bundle; retain strict canonical HTTP checks and compare deployed homepage notice as well as policy. Add a one-attempt option for bounded operational checks.
4. scripts/publish_search_assets.py: validate source privacy files before modifying deployment assets. Fail before publication on broken/missing policy.
5. tests/test_search_publication.py: negative integration check that invalid source cannot replace the target's existing files; existing publication tests cover preservation of forecast records.
6. Commit research/plan and implementation in isolated branch; run focused tests and browser layout/link checks; create PR, check CI, merge and observe existing search publication workflow. Respect provider limits.
7. Fetch public URLs after successful deployment, compare expected content, verify link navigation in browser. Record production evidence. Report website result separately from Google review status.

Trade-off: retain existing long-form detailed policy while adding a compact readable summary; no new login, tracking, contact collection or prediction changes. Source-only edits are not considered deployment success.

Validation repair: fix the existing archive test fixture's ZIP timestamp, which otherwise produces inconsistent checksums at wall-clock boundaries. No production forecasting or collector logic is changed.
