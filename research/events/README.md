# ETH hourly event research beta

The paid-service objective is detecting useful ETH price, volatility and trend
changes. This release builds the measurement and operation needed to test that
objective. It does **not** establish a predictive edge or activate subscriptions.

## Frozen definitions

- Canonical price: Coinbase Exchange ETH-USD; BTC-USD is a past-only covariate.
  Binance ETH-USDT daily history is not substituted for this instrument.
- Horizons: 6h, 24h, 72h, 168h, 336h, 720h. Independent models per horizon.
- Input: closed hourly OHLCV, a complete 721-bar window, no forward/backward fill.
  Features use returns, realized volatility, volume, drawdown, ETH/BTC relationships,
  past trend and hour of day. Actual receipt time is distinct from candle close.
- Each forecast freezes its reference close and UTC input cutoff. To avoid counting
  events that happened before publication, the event window starts at the **next
  full UTC hour** and lasts the stated horizon. Terminal target is that window's
  end. The horizon does not misleadingly include the partially observed issue hour.
- Barrier = max(log(1 + floor), trailing 720-hour log-return std × sqrt(h)). Floors
  are 1.5/3/5/7/10/15% by horizon. Log-symmetric down thresholds differ from simple
  symmetric percentages. The UI shows actual upper/lower prices.
- Up/down barrier hits are separate binary events; both may happen. A terminal
  down/flat/up class is another target. OHLC cannot order two hits within one bar.
- CatBoost has direct event and direction objectives, plus 10/50/90% price
  quantiles. The interval is nominal, and its realized coverage is reported.

## Evaluation and efficient selection

`signal_pipeline/protocol.py` is the versioned protocol. Every monthly outer fit
has a separate earlier validation period and label-end purge plus one-hour embargo.
For horizons under 14 days, training uses up to 730 days before the separate 90-day
validation window; longer horizons use up to 1,095 days and separate 365-day
validation. At least 700 training and 120 validation origins are required.
Training samples every six hours; full eligible historical evaluation samples one
origin per UTC day. Initial learning periods and unavailable windows are excluded
and coverage/first-last origins are reported, rather than manufacturing predictions.

Candidates are smoothed event frequencies, scaled logistic models, CatBoost, and
a 50/50 CatBoost-frequency probability blend. Only earlier validation Brier scores
choose the model/blend, followed by a fixed-architecture refit on matured observations.
Each candidate's alert threshold is fixed independently from its validation
negatives (5% false-positive target; strict greater-than handles constant ties).
This is an operating point, not a promised subsequent false-positive rate.

The report compares identical origins: event/terminal Brier, log loss, PR-AUC,
recall, false positives, interval coverage and width, price MAE versus no change,
yearly breakdowns and a paired calendar-block interval. Overlapping horizons are
not treated as independent observations. Historical source receipt vintages are
not reconstructable: backfilled history is explicitly retrospective development.
Foundation-model pretraining overlap creates an additional limitation.

An hourly run **cannot train**. It verifies a hash-pinned active monthly checkpoint,
uses it for inference, settles mature outcomes, and reports stale/missing inputs.
Monthly fits are cached using training implementation, protocol and source hashes.
Changing unrelated UI or publication code does not retrain the neural/tree models.

## Durable issuance and updates

| Work | Schedule (UTC) | Behavior |
|---|---|---|
| Closed bars / inference / settlement | Every hour :08 | Two price feeds; bounded 90s collection, no in-job training |
| Independent site check | Every hour :38 | Check release and age, dispatch at most one recovery if workers idle |
| Research review / replay | Sunday 07:13 | Reuse valid checkpoints; retain measured results and source errors |
| New monthly checkpoints | First day 00:23 | Purged selection and matured refit; current old month cannot issue as new |
| Rolling prospective report | Each inference | 28/84/365-day matured records; no automatic promotion |
| Consistent backup | UTC 00 hour and new research release | Online SQLite backup and integrity check |

GitHub schedules may be delayed; this is a research beta scheduler, not a paid
service uptime SLA. Actual source/input/issue/publication times appear in the site.

1. Collect only documented Coinbase candles. Retry at most three times under the
   budget, respect Retry-After, stop on denied access. Never route around denial.
2. Save raw response hashes, original receipt, and append-only source revisions.
   Incomplete/missing bars block affected features or truth settlement.
3. Persist forecast and outbox atomically. Same slot/horizon/model/definition yields
   the original payload. Historical replay is never inserted into actual issuance.
4. Preserve the issuance SQLite ledger on `data/event-ledger` before publication.
   Raw candles and large historical models are not committed to that branch.
5. Save bounded hourly state artifacts (40-day inputs, active models, two-day
   retention), then publish public research JSON and charts via `data/daily-forecast`.
   Research artifacts retain full inputs/checkpoints for 30 days, renewed weekly;
   daily consistent snapshots retain 30 days. Long-term external object storage is
   still needed before claiming the production retention/RPO goals in the plan.
6. Verify the external release and issued IDs. Only then append a publication
   receipt. Unpublished/late-published predictions remain auditable but are excluded
   from timely prospective performance; previous public forecasts are never erased.

The authoritative actual-issued ledger is restored separately from research state;
a research replay or backup must never overwrite it. Lost/expired input artifacts
can be reconstructed from sources/research; actual-issued records are restored from
their durable branch. Source corrections create new outcome revisions, preserving
the original forecast and earlier truth versions.

## Commands

```bash
python -m pip install -r requirements-events.txt
python scripts/run_event_forecast.py --backfill --replay --horizons 6 24 72 168 336 720
python scripts/event_state.py merge lake/research-source lake/signals
python scripts/run_event_forecast.py --collect --daily --review --horizons 6 24 72 168 336 720
python scripts/event_state.py snapshot lake/signals consistent-backup
python scripts/verify_event_site.py --require-ready
python -m pytest tests -q
```

## Service readiness and next gates

This public beta deliberately contains only research forecasts. No paid claim,
subscription charge, private API key or personal subscriber data is published.
The earlier daily CatBoost + patch Transformer is retired from automated issuance.
Its daily settlement continues; its full replay is manual-only and charts are archived.
The following require further evidence or provider configuration, and must not be
represented as deployed production features:

- Predictive superiority on untouched prospective data, independently grouped event
  counts, calibrated alert burden, downside behavior and realistic latency/costs.
- Additional OI/funding/options/ETF data **after** availability/licensing validation
  and incremental-value comparisons. Unavailable historical features are not zero-filled.
- An always-on worker, external heartbeat, object storage/production database,
  measured restore drill and recovery targets for paid uptime.
- Server-authorized subscriber access, customer-selected notification destination,
  and a configured payment provider/account with test-mode webhook/cancellation
  validation. Static public JSON cannot be made private by hiding it in JavaScript.

Reference APIs: [Coinbase candles](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles),
[GitHub schedules](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
