# ForecastBundleV2 — R1 architecture contract

Status: **offline contract only; no production promotion**.

Roadmap authority: issue #68. Promotion gates: issue #56.

## Why this exists

The incumbent event pipeline selects one model family using a combined terminal/path Brier objective and then carries that family's return quantiles. That couples event classification and distribution outputs even though their correct objectives differ.

R1 breaks that coupling before any new model is promoted.

## Independent heads

| Head | Output | Primary objective | Mandatory baseline |
|---|---|---|---|
| Center | central/median log return | MAE / median pinball | no-change / zero return |
| Volatility | future realized variance / scale | QLIKE | persistence / simple HAR baseline |
| Distribution | coherent return quantiles / intervals | WIS, pinball, coverage, width | simple calibrated distribution + incumbent |
| Event | barrier/path/terminal probabilities | Brier, log-loss, ECE | prior frequency / climatology |

A head can only be promoted using its own frozen objective and gates. Improvement in another head is not transferable evidence.

## Frozen horizon semantics

The current protocol is preserved in R1:

- reference price = ETH close at `input_cutoff`;
- first future bar starts after the origin;
- endpoint return uses the frozen `h+1` implementation;
- `target_end = input_cutoff + (horizon_hours + 1) hours`;
- supported horizons remain 6 / 24 / 72 / 168 / 336 / 720 hours.

Changing these semantics requires a new experiment/protocol family, not an in-place V2 patch.

## PIT / provenance fields

Every `ForecastBundleV2` carries:

- issue time;
- input cutoff;
- latest source `available_at`;
- target end;
- reference price;
- immutable input snapshot identifier;
- feature version;
- head-specific model decisions.

The contract rejects sources whose `available_at` is later than the issue time.

## Baseline fallback

Every head has an explicit baseline. A candidate that fails its head's required relative-improvement threshold falls back to that baseline. This allows, for example, volatility to improve while Center remains at no-change.

## Current R1 scope

This PR adds only the contract and invariant tests. It does **not**:

- import the contract from incumbent live inference;
- alter `models.py` selection;
- change checkpoints;
- publish new forecasts;
- change site output;
- claim predictive edge.

## Next R1 increments

1. Add head-specific evaluator interfaces.
2. Add an incumbent compatibility adapter that converts current research outputs into a V2 bundle without changing live behavior.
3. Add offline replay that reports each head separately.
4. Only after R1 invariants pass, begin R2 volatility/range challengers for 6h / 24h / 72h.
