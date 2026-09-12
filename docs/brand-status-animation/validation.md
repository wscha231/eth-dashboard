# Implementation validation — 2026-09-12 UTC

The user approved implementation and the additional donation section. The original
image remains embedded unchanged; only SVG light/rain layers and state control were
added. The header and status logo use the same resolved state.

## Completed checks

- `python -m pytest tests -q`: **312 passed**. Existing frontend tests now load
  both production scripts and assert the approved connection-error precedence.
  Their immutable issued-price and prior-forecast checks remain intact.
- Chromium 153 / Playwright, using a fixed clock and six valid fixture forecasts:
  verified actual changing circuit dash offsets, operating → maintenance →
  operating, failed status during maintenance, forecast connection failure,
  incomplete horizons, expiry and recovery, initially failed requests, unknown
  status, reduced motion and hidden-tab pausing. Badge and logo always agreed.
- Maintenance was processed while the replay request remained pending.
- Maintenance hides the effects, pauses every animation and dims the raster to
  brightness 0.26 / saturation 0.25. Inspected operating and maintenance screenshots.
- A status message containing HTML rendered as literal text.
- Desktop 1365px and mobile 320px/390px inspected. No document overflow or page
  errors with external font/chart services blocked. This check found a pre-existing
  unguarded legacy Chart call and default canvas overflow; added a Chart-availability
  guard and max-width for chart canvases. Korean glyphs were unavailable in the
  test browser's font bundle; source Korean text is retained unchanged.
- Publication test copied/staged all owned assets, including both SVGs and the
  status configuration, while preserving forecast JSON, donation wallets and
  unrelated verification files. Hybrid and hourly use this same asset publisher.
- JavaScript syntax, shell syntax, SVG structure and `git diff --check` passed.

Optional reproducible browser check: install Playwright and its Chromium browser,
then run `node scripts/test_brand_browser.cjs`. `CHROMIUM_EXECUTABLE` and
`CHROMIUM_ARGS` (JSON array) allow a supplied browser; `BRAND_QA_OUTPUT` controls
the temporary screenshot directory. All forecast fixtures stay in the test runner;
no synthetic prices are written to published JSON.

## Publication boundary

`python scripts/deployment_policy.py --require-enabled` returned **75**, confirming
the existing provider cooldown until **2026-09-13 20:10 UTC / 2026-09-14 05:10 KST**.
No early deployment is authorized by this implementation. Main / data-branch
updates do not establish that the public site changed. Public URL verification
must happen after the next eligible successful deployment.

## Research boundary

Official prices are linked next to the figures in the page and research notes.
The $4,826 candidate budget is $3,588 + $1,238, excluding optional commercial
quotes and other operating costs. The 3% MAE / 5% Brier thresholds are labelled
proposed pilot targets with illustrative arithmetic, not measured or expected
forecast improvement. No paid API was purchased or newly integrated.

## User correction: place the actual animation at the top

The top panel now contains the original **MP4 video**, at 240px wide on desktop
and 192px on mobile, with the service status beside/below it. The earlier 112px
SVG mount is no longer used in that panel. SHA-256 of the copied 5-second video
matches the file shown to the user:
`e6294ea238ecfebb5568b7a2f1ddde4631b8d2eec414ef018cb1c83082385871`.

Chromium verified real `currentTime` advancement, looping over the 5-second end,
manual pause/resume, maintenance stopping time advancement, automatic recovery,
reduced-motion / hidden-tab pausing, blocked-playback recovery with a button,
and a missing MP4 displaying a static fallback with a visible load-error note.
All prior service-state cases and 320px/390px overflow checks passed. No page
errors. The published source uses a native muted/loop/playsinline video and the
same service-state resolver. The MP4 is included in the shared asset publisher.

The full Python suite passed **319 tests** on the updated main baseline. After
final markup and responsive-control adjustments, **17 relevant tests** passed
again. JavaScript syntax and `git diff --check` passed. The viewport check also
constrained the newly lengthened Evidence selector's grid track to prevent
mobile overflow; its options and data remain unchanged.

`top-animation-preview.png` shows the rendered top panel with fixture status;
it is a placement preview, not evidence of deployment or a real current forecast.
The previously recorded Vercel cooldown still applies to public deployment.
