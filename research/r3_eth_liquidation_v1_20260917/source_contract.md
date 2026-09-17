# R3 P0-4 ETH liquidation / cascade source contract v1

Frozen: 2026-09-17

## Purpose

Collect prospective ETH perpetual liquidation receipts for later causal research. This source is **not** authorized for MODEL, website or public redistribution by this contract.

## Primary source

- Provider: Bybit V5 public WebSocket
- Endpoint: `wss://stream.bybit.com/v5/public/linear`
- Topic: `allLiquidation.ETHUSDT`
- Symbol: `ETHUSDT`
- Source documentation states that this topic pushes all liquidation events for the subscribed symbol and uses a 500 ms push frequency.

## PIT / availability contract

- `event_time`: exchange-provided liquidation update timestamp `T`.
- `stream_generated_at`: Bybit message timestamp `ts`.
- `received_at`: actual UTC receipt time at EtherForecast.
- `available_at = received_at`.
- No event is assigned an availability time earlier than the first EtherForecast receipt.
- Capture gaps are retained in `capture_sessions.jsonl` and are never filled synthetically.

## History contract

`prospective_receipts_only`.

There is no historical market-liquidation backfill in this implementation. Public recent trades, user liquidation records, reconstructed candles, or inferred liquidations must not be substituted for missing market liquidation events.

## Source semantics

Fields preserved from the official stream:
- `T` -> event time;
- `s` -> symbol;
- `S` -> source side;
- `v` -> executed size;
- `p` -> bankruptcy price.

Bybit documents `S=Buy` as a long position liquidation and `S=Sell` as a short position liquidation. EtherForecast stores that documented position label without adding a trading interpretation.

`notional_usdt = size_eth * bankruptcy_price_usdt` is a transparent unit conversion only. It is marked as derived and is not a tuned feature.

## Rights boundary

- `license_status = rights_unreviewed_public_stream`
- `public_redistribution = blocked_pending_rights_review`
- `model_use = blocked_until_rights_prospective_continuity_and_preregistered_ablation_pass`
- `site_use = blocked`

Public API/WebSocket availability is not treated as permission for commercial redistribution.

## Collection cadence

- Pull requests / pushes: short connectivity and contract validation window only.
- Scheduled main collection: bounded ~55 minute observation window each hour.
- Every session records actual start/end, connection count, disconnects, parse errors, acknowledgement state, duration and event count.
- Delayed GitHub Actions scheduling can create gaps. Those gaps are evidence and must remain visible.

## Persistence

The append-only logical state is preserved in the `eth-liquidation-research-state` GitHub Actions artifact and registered with the existing private Google Drive R3 archive stream.

## Promotion boundary

This data family cannot change Center, Volatility, Distribution or Event outputs merely because collection succeeds. A future MODEL experiment requires:
1. sufficient prospective continuity and explicit gap accounting;
2. source-rights review;
3. a preregistered liquidation/cascade hypothesis;
4. matched PIT-safe replay or prospective-only evaluation as applicable;
5. the frozen task-specific promotion gates;
6. manual review.
