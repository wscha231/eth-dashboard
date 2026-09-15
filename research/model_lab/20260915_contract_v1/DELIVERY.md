# Delivery locations — 2026-09-15

This is an offline MODEL checkpoint. Read the PR conversation for later CI and readback receipts.
Core source and tests in this branch are the exact locally tested bytes; contract/report documents
and the data-request queue provide a shared handoff, not automatic cross-chat message delivery.

Private Drive root: EtherForecast-Storage (existing connection; no sharing permissions changed).

- Code/results: `1SdBB6hxTS07Mc776xaKaj-L-aqEvTdFl`, MODEL-code-results-20260915-709f17962d32.zip, 3747249 bytes; SHA256 `709f17962d3200c6a7f2b874a4fb6aa34400cc54ba04618f1768e5d848b9a16b`.
- Inputs: `14LIxtO7lOKtxWoav6UlI4yoCfSWDuocT`, MODEL-pinned-inputs-20260915-67b67e3c357a.zip, 19321767 bytes; SHA256 `67b67e3c357a675c73172eb937ba98e99dbf628b9c9b1c9f2c41fe604c4843cd`.
- Input release SHA256 `4fac29b602ebccf4559ffeb889113afb82449df6d1a10b461ccc0a9b4f613b11`.

Both native Drive uploads succeeded. Upload alone is not a readback or restoration certificate;
check the appended PR delivery receipt. No raw market dataset/model binary is committed publicly.

Download each Drive ZIP with the connected Drive tool, verify its full SHA256, extract into a
NEW empty directory, and use the contained release plus the commands in MODEL_STATUS.md.
The evidence bundle contains row-level predictions/actuals, year/regime summaries, frozen policy,
model hashes, fitted CatBoost/Ridge/HAR artifacts and focused test output. The separate direction
model is only exported as probabilities, not a fitted live inference artifact.

The supplied input ZIP is a MODEL transport subset of existing archived source bytes, not a
second collector/archive engine and not a new authoritative DATA release. The original Drive
research stream's two SQLite backups still require a separate full logical restoration audit.

No merge, deployment, promotion, trading, paid source activation, source deletion or schedule
change is authorized by this evidence. Monthly formal walk-forward and independent prospective
validation remain future integration work. The annual development benchmark is not the monthly
production service and not an untouched test period.
