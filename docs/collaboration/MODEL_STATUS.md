# MODEL_STATUS — 2026-09-15

Status: offline implementation and historical development checkpoint. NOT production-complete.

Code: model_lab/contracts.py, reproduce.py, benchmark.py; tests/test_model_lab.py.
Contract: docs/collaboration/ETH_DATA_MODEL_CONTRACT.md (draft for DATA review).
Report: research/model_lab/20260915_contract_v1/REPORT.md. Requests: requests/eth_model_data_requests_20260915.jsonl.

93 focused local tests passed; actual six-horizon source/target and archived metrics reproduced.
Three approach families plus fixed ablation were run on matched 2024–2026 daily origins.
No convincing new price edge; HAR variance candidate only, all promotion disabled.

Upstream SHA: e954a2534743e692a048d5b40738c4cbf3253938. Release SHA256: 4fac29b602ebccf4559ffeb889113afb82449df6d1a10b461ccc0a9b4f613b11.
DATA shared function fingerprint is checked, and each file in the input receipt is checksum-pinned.
The DATA 20260915 handoff package was not located; do not treat its reported pilot as reproduced.

Full repository CI, branch/PR IDs, actual private Drive object IDs and readback hash receipts
must be checked in the delivery record. Do not infer them from this pre-publication checkpoint.

Resume after reading the report, current main/PR and DATA release; do not merge or deploy:
1. Obtain approved DATA release/registry and requested source availability metadata.
2. Use the saved input package or verified existing Drive restoration, never an expired sandbox URL.
3. Run the commands below into NEW output directories; compare release and source hashes.
4. Pre-register monthly walk-forward/calibration and independent prospective evidence requirements.
5. Only reviewed candidates may later reuse the existing immutable ledger and public-feed workflow.

```bash
python -m pytest tests/test_model_lab.py -q
python -m model_lab.reproduce --release INPUT/release.json --release-sha256 4fac29b602ebccf4559ffeb889113afb82449df6d1a10b461ccc0a9b4f613b11 --input-root INPUT --output RUNS/new-reproduction
python -m model_lab.benchmark --release INPUT/release.json --release-sha256 4fac29b602ebccf4559ffeb889113afb82449df6d1a10b461ccc0a9b4f613b11 --input-root INPUT --output RUNS/new-benchmark
```

No automatic cross-chat message delivery is assumed. DATA reads this shared status/requests file.
