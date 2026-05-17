# Data directory overview

This folder holds the data artifacts produced and consumed by Phase 1 ingestion.

Structure
---------

- `data/raw/` — Source-specific raw Parquet files saved after ingestion.
- `data/interim/` — Cleaner/interim files created during processing (not final).
- `data/processed/` — Final canonical datasets used for analysis and modelling.
- `data/broker_cache/` — Optional broker-side cache (not committed).

Canonical processed datasets (Phase 1)
-----------------------------------

- `data/processed/us_vix.parquet` — US implied-volatility panel (CBOE/FRED/Yahoo sources).
- `data/processed/us_underlying.parquet` — US underlying OHLCV (SPX/SPY from Yahoo).
- `data/processed/india_vix.parquet` — India implied-volatility panel (NSE/Yahoo sources).
- `data/processed/india_underlying.parquet` — India underlying OHLCV (NIFTY from Yahoo/NSE).

Per-source raw paths are configured in `configs/data_sources.yaml` under each
`source_id` as `raw_path` (for example `data/raw/us_vix_cboe.parquet`).

Notes
-----

- Do not commit large raw or broker cache files to version control.
- If a source requires manual CSV download (CBOE/NSE), place files under `data/manual/` and
  use the `--local-csv` override with `scripts/download_data.py`.
- For a quick audit of ingested datasets and validation results, see:

  `notebooks/01.ipynb`
