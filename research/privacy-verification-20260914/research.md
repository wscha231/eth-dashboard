# OAuth privacy verification investigation — 2026-09-14

User requests repair and production publication of the three Google branding findings.

## Evidence
- Source main: ab3fcf6ec962b34bdbeb51d088fc4d7ebae7a1a1. No AGENTS.md in checkout.
- Direct unauthenticated HTTPS GETs on 2026-09-14 returned HTTP 200 text/html for https://etherforecast.live/ and https://etherforecast.live/privacy.html without redirection. Downloaded privacy bytes equal main's public/privacy.html. These findings do not establish Google approval or historical availability.
- Homepage policy anchors exist only in the footer after a long dashboard and donation section. Upper navigation lacks a policy link.
- Policy includes data categories, planned drive.file access, purpose, providers, security, retention, deletion, Limited Use and contact. It accurately marks operator Drive backup as planned; no implemented OAuth backup client was found. Do not claim active integration or invent support email/console scopes.
- Brand at top is EtherForecast; legacy page title and notice use ETH Forecast. Clarify these are the same service.
- Vercel serves forecast_site/public from data/daily-forecast, not main. Git deployment enabled only for that data branch. Cooldown not_before=2026-09-13T20:10:00Z has passed. Common publisher copies homepage, privacy, animation assets and Vercel config; latest hourly runs succeeded.
- Existing verifier checks policy byte identity after publication. Common publisher has no mandatory pre-publication privacy validation. A future invalid source can replace the valid production page before failure is detected.

## Official requirements
https://support.google.com/cloud/answer/13807376?hl=en — public homepage, app purpose and Google data purpose, verified domain and accessible privacy link matching consent screen.
https://support.google.com/cloud/answer/13806988?hl=en — comprehensive disclosure of Google data access, use, storage and sharing.

## Limits
PR validation found an existing wall-clock-sensitive fixture in tests/test_forward_research.py: archive() recreates ZIP bytes independently for payload and checksum requests, with an implicit current timestamp. Reproduced that identical CSV bytes with timestamps two seconds apart yield different SHA-256 values. Fix only the test ZIP metadata to a fixed timestamp; production collection and checksum validation remain intact.

Google Cloud consent-screen configuration and review state were not inspected. Previous findings can remain until re-review; a website repair alone does not update reviewer status. No invented claim about the root cause of the earlier review.
