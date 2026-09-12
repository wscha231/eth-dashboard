# Execution and follow-up

PR #38 merged at `73c53976a7c7562353f27e9bc3a66131386a5f1e` after 298 tests passed locally and in GitHub CI run 34715371982. Production daily run 34715476316 completed successfully at 2026-09-12 19:55 UTC and preserved 15 normalized market/vendor CSV caches. The actual archived source partitions restored exactly to 1,918 latest source rows with original receipt times, values and hashes (1,920 source versions in the snapshot).

Production verification revealed two additional concrete issues:

1. A missing vendor measurement serialized as `NaN` in collector_summary.json. Strict JSON consumers cannot parse it. The collector now writes null for nonfinite/missing values, retaining the source error and keeping valid numeric measurements unchanged. Apply to the summary, quality audit and optional JSON stdout.
2. The merge's simultaneous daily/source/hourly triggers replaced pending workers under the existing shared publication concurrency group. The daily collector succeeded, but hourly was cancelled before any job started. Chain hourly after successful `Daily archived hybrid settlement`, the existing final daily writer. This adds no cycle: hourly completion only starts the watchdog, not the daily chain. The normal hourly schedule and idle/cooldown/deadline recovery controls remain. Source collector code/cache changes also explicitly trigger the daily workflow.

The daily run reports Binance HTTP 451 and a CoinGecko DeFi HTTP 429; these are missing auxiliary sources, not successful collection. No regional access bypass or synthetic filling was attempted. The current hourly model uses Coinbase bars independently.

The follow-up keeps model values, promotion gates and original records intact. Validate strict JSON handling of nested missing values, retain error explanations, rerun the required regression CI, and verify the successful daily chain then hourly archive/publication before claiming deployment completion.
