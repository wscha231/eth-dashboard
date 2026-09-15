# Free-source D003/D004 implementation plan

1. Add `data_lab/free_sources.py`.
   - Bybit funding backwards pagination.
   - Bybit OI bounded-date + cursor pagination.
   - Bybit premium-index daily history.
   - Past-only deterministic derivative features.
   - FRED/ALFRED initial-release parser and conservative as-of daily state.
   - Availability sidecars and coverage helpers.

2. Add `scripts/collect_free_research_sources.py`.
   - First run backfills from 2017 where the provider has data.
   - Subsequent runs refresh an overlap tail when a historical cache already reaches the requested start.
   - Reuse existing DefiLlama collectors for a full-history cache repair.
   - Write `lake/reports/free_source_backfill.json`.

3. Wire the script into `.github/workflows/daily_forecast.yml`.
   - Restore old daily cache first.
   - Refresh free sources.
   - Pass available Bybit and FRED as-of files to the existing daily collector through `--manual-features-csv`.
   - Persist vendor CSVs through the existing `daily_source_state.py`.
   - Preserve all data through the existing daily Drive snapshot.

4. Tests.
   - Pagination progress/loop guard.
   - Unit-preserving OI.
   - Premium parsing.
   - Next-day availability.
   - FRED output_type=4 and release-date timing.
   - No future release available on release date itself.

5. Promotion rule.
   Data collection may be merged when tests and one real workflow run pass. These features remain research inputs. MODEL must still perform matched-origin ablation before any forecasting promotion.
