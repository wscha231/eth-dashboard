#!/usr/bin/env bash
set -euo pipefail
# Call only while holding the shared daily-forecast GitHub concurrency group.
git config user.name "eth-forecast-bot"
git config user.email "eth-forecast-bot@users.noreply.github.com"
git fetch origin data/daily-forecast:refs/remotes/origin/data/daily-forecast
if git ls-remote --exit-code --heads origin data/hybrid-forecast > /dev/null; then
  git fetch origin data/hybrid-forecast:refs/remotes/origin/data/hybrid-forecast
  hybrid_base=origin/data/hybrid-forecast
else
  hybrid_base=origin/data/daily-forecast
fi
git worktree add --detach "$RUNNER_TEMP/hybrid-ledger" "$hybrid_base"
mkdir -p "$RUNNER_TEMP/hybrid-ledger/lake/hybrid"
cp -a lake/hybrid/. "$RUNNER_TEMP/hybrid-ledger/lake/hybrid/"
git -C "$RUNNER_TEMP/hybrid-ledger" add -f lake/hybrid
if ! git -C "$RUNNER_TEMP/hybrid-ledger" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/hybrid-ledger" commit -m "chore(hybrid): persist monthly checkpoints and immutable issued record"
  git -C "$RUNNER_TEMP/hybrid-ledger" push origin HEAD:refs/heads/data/hybrid-forecast
fi
# Checkpoints may be retained on failure, but incomplete charts are never published.
if [ "${HYBRID_PUBLISH_COMPLETE:-false}" != "true" ]; then exit 0; fi
git worktree add --detach "$RUNNER_TEMP/hybrid-site" origin/data/daily-forecast
cp lake/hybrid/hybrid_forecast.json "$RUNNER_TEMP/hybrid-site/forecast_site/public/hybrid_forecast.json"
cp lake/hybrid/hybrid_predictions.csv.gz "$RUNNER_TEMP/hybrid-site/forecast_site/public/hybrid_predictions.csv.gz"
cp forecast_site/public/index.html "$RUNNER_TEMP/hybrid-site/forecast_site/public/index.html"
git -C "$RUNNER_TEMP/hybrid-site" add -f forecast_site/public/hybrid_forecast.json forecast_site/public/hybrid_predictions.csv.gz forecast_site/public/index.html
if ! git -C "$RUNNER_TEMP/hybrid-site" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/hybrid-site" commit -m "chore(site): publish optimized CatBoost + Transformer forecasts"
  git -C "$RUNNER_TEMP/hybrid-site" push origin HEAD:refs/heads/data/daily-forecast
fi
python scripts/verify_hybrid_site.py --expected lake/hybrid/hybrid_forecast.json
