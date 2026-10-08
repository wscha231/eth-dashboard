# Open Standard OUSD chain-allocation research source contract

Updated: 2026-10-08

## Scope

This contract covers the research-only prospective receipt
`lake/raw/vendor/open_standard_ousd_chain_supply_receipt.csv`.

It does not authorize MODEL use, forecast publication, site display, public API
redistribution, paid-product use, threshold changes, or promotion.

## Identity

The tracked asset is **Open USD (OUSD)** issued by Bridge Building Inc. for Open
Standard, not Origin Dollar (which also uses ticker OUSD).

Official launch-chain identities published by Open Standard on 2026-09-30:

- Base: `0xB2000000000000000000002fEb517dFeC7415344`
- Ethereum: `0x9f6F3991D525015a6F8CaF062C83b62fD3AC4436`
- Solana: `ousd2mJsPEckLHcSCDxyKD7NDGARZcfLbDZkKiatYHB`
- Tempo: `0x20c0000000000000000000006a37DA5C996874BE`

The collector uses the already-used DefiLlama public stablecoin endpoint as a
transport/source and matches exact name `Open USD`, exact symbol `OUSD`,
`peggedUSD`, and all four launch chains. A missing launch chain or any new
positive chain supply fails closed pending contract review.

## Time semantics

- `observation_time`: unknown; the current DefiLlama snapshot does not expose it.
- `public_available_at`: not separately published by this endpoint.
- `received_at`: actual UTC receipt time recorded by EtherForecast.
- `available_at`: conservatively set equal to `received_at`.

No receipt before the collector is deployed may be reconstructed or relabeled as
prospective evidence. Missing collection is missing; it is never zero-filled,
forward-filled, interpolated, or assigned today's date.

## Storage and rights boundary

The source is collected through the existing free-research step in
`daily_forecast.yml`. The raw OUSD receipt path is explicitly blocked from the
public hot data branches. It remains eligible for the existing
`daily-source-state` Actions artifact and daily-data Drive preservation path.

DefiLlama is already an active EtherForecast research source, but this contract
does **not** infer or expand public-redistribution, site, API, or commercial-use
rights. Those uses remain blocked pending a separate rights review.

## Research use

Two distinct evidence classes remain separate:

1. DefiLlama global/Ethereum/Base historical stablecoin series:
   reconstructed/development research only.
2. Open Standard OUSD chain-allocation receipt:
   prospective receipt only from actual post-deployment fetches.

The first frozen 24h Center feature family is:

- `sc_global_growth_1d`
- `sc_ethbase_share_delta_1d`
- `sc_ethbase_share_delta_7d`
- `sc_ethbase_vs_rest_flow_1d`

OUSD-specific receipts are **not** added to that historical model screen because
OUSD did not exist historically. They accumulate separately for later
prospective analysis.

## Failure policy

Source timeout, malformed values, boolean/non-numeric supply, negative supply,
ambiguous identity, missing launch chains, unreviewed positive chains, or a
material chain-sum/total conflict makes the OUSD receipt unavailable for that
run. Such a failure must not stop a valid baseline forecast or be replaced with
a synthetic value.
