"""LLM-based market analyst experiment.

This module is intentionally decoupled from ``eth_price_forecast`` and the
``forecast_site`` tree. It explores a different product framing: instead of
producing a single-point regression forecast (which OOF diagnostics show has
negative skill vs naive at h=30), we ask a language model to synthesise a
*market-environment view* from:

  - quantitative snapshot: recent prices, on-chain deltas, macro deltas
  - qualitative context: news headlines, social sentiment, regulatory events

and output a **structured view**: direction, confidence, thesis, key risks,
and "what would change the view".

Lives beside the existing stack — no imports from or into it. If the POC
shows skill, we'll wire it into the site as a complementary "Analyst view"
panel alongside (or replacing) the regression point forecast.

Modules:
  news_store.py  : SQLite schema for headline storage
  news_fetcher.py: RSS pull from free crypto news sources
  snapshot.py    : builds the numeric portion of the analyst prompt from
                   lake/gold/eth_master_daily.csv, date-gated for leak-safe
                   historical evaluation
  analyst.py     : prompt assembly + Anthropic API call + JSON parsing
  backtest.py    : evaluate N historical dates, compare skill vs naive

Status: experiment — not yet deployed, not yet wired to site.
"""
