# DATA checkpoint — 2026-09-15

## Delivered, not promoted
Code commit: `db61e37bab17ce89c2fd661589c72109c4ed6e23`.
Branch: `data/release-feature-contract-20260915`, additive to main
`e954a2534743e692a048d5b40738c4cbf3253938`. MODEL PR53 stays separate.

Authoritative DATA research release: `eth-data-18525a7b61ee33af20467989`.
Release SHA256: `921d8e150296ce1c9fc52ab699eb8b171870efe80ad2845f4674a2f057e7c7ef`.
Archive SHA256: `6e40f8d8bd31788358f169849b55688cedd6346e580e55cb854592e9820176db`.
Data cutoff: 2026-09-14 07:00 UTC; generated 2026-09-15 09:04:37.848003 UTC.
These are pinned historical inputs, not a new live market update.

Private Drive release (37,788,941 bytes):
https://drive.google.com/file/d/1xK14XVNdTc4yN7v2w2Fj38TeUyIN_z9U/view?usp=drivesdk

Original missing research handoff restored (1,745,003 bytes):
https://drive.google.com/file/d/1oDIdu-PCHROi8qhy1AaMH_bsHLW7iM_M/view?usp=drivesdk
SHA256: `802bdaeea4cbab6052e66e3a7210414967d0ca9cdcbe5fd62c0e4c500aac6faf`.
Both remain in the existing private EtherForecast-Storage root. No sharing changes.

## Completed verification
- 61 focused DATA tests passed locally (pytest 9.0.2).
- Both Drive uploads downloaded again; complete SHA256 equality.
- New DATA ZIP: 11 files, 10 manifest-pinned partitions verified, SQLite integrity OK.
- Original handoff: all23 manifested files verified, not recreated from memory.
- Input DB: ETH85,001 / BTC84,986 bars; common hourly grid85,039.
  Missing38 /53; longest gaps5h /15h respectively. Missing source reasons remain unknown.
- Incumbent32 features reproduced exactly through actual MODEL PR53 consumer.
- New18 hourly extension columns share the same time grid; each has a receipt timestamp.
- All6 horizons passed source/target and archived metric reproduction after Drive readback.
- Candidate metric CSV is byte-identical to the prior MODEL delivery.
- Source files unchanged; prospective authorization correctly rejected this research release.

Local integration uses the actual MODEL delivery code plus a fingerprint-verified
connector-read upstream function extract, not a full repository checkout. Full CI
results must be verified separately on the PR, not inferred from local tests.

## Important remaining limits
quality=partial; vintage=reconstructed; rights=unreviewed_no_public_redistribution.
This delivery verifies **its own** archive, not all original ef1 Drive archives.
The original separately backed-up SQLite comparison remains open. Original historical
publication/receipt vintages, live latency SLA and source rights are not invented.

The upstream float rolling timestamp differs from an exact receipt maximum by up to
128ns in either direction. Exact sidecar supplied; original formulas are unchanged.
This tiny precision issue is not evidence explaining poor investment predictions.

No model training, performance improvement claim, promotion, production change,
merge, site deployment, paid API use, source deletion or new continuous collector.
Funding/OI/basis and macro/onchain genuine-history acquisition remain open.
See `requests/eth_data_responses_20260915.jsonl` for per-request acceptance status.

## MODEL continuation
Review the additive DATA commit on the MODEL branch without merging production.
Restore the private release ZIP into a fresh directory and verify its SHA256.
Use existing `model_lab.contracts.PinnedRelease/load_bars/consume_features` unchanged.
Use `data_lab.release.load_feature_snapshot(..., family="extensions")` for optional
new columns. Exclude `__available_at` columns from predictors. Historical receipt
timestamps must not be moved into the past. Evaluate added features on matched
origins using the existing target spec; keep original daily pilot separate.

```bash
python -m data_lab verify --root RESTORED_DATA_RELEASE --sha256 921d8e150296ce1c9fc52ab699eb8b171870efe80ad2845f4674a2f057e7c7ef
python -m model_lab.reproduce --release RESTORED_DATA_RELEASE/release.json --release-sha256 921d8e150296ce1c9fc52ab699eb8b171870efe80ad2845f4674a2f057e7c7ef --input-root RESTORED_DATA_RELEASE --output NEW_MODEL_OUTPUT
```

Return acceptance or precise additional DATA requests; do not mark predictive edge
from data readiness. This shared document is not automatic cross-chat delivery.
