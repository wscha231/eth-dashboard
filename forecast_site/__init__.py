"""ETH forecast public service: persistence + cron glue.

Modules:
- db             : SQLite connection + schema initializer.
- persist_forecast: take latest_forecast_summary.csv and upsert into DB.
- backfill_actuals: resolve past forecasts to realized ETH closes + update
                   accuracy snapshots.

Run order on each daily cron tick:
    1. eth_data_collector (build master dataset CSV)
    2. eth_price_forecast.run_suite (produces latest_forecast_summary.csv)
    3. forecast_site.persist_forecast (CSV -> DB)
    4. forecast_site.backfill_actuals (resolve anything whose target date has passed)
"""
