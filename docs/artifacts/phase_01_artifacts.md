# Phase 1 Artifacts

## Status

Complete / frozen.

Phase 1 artifacts cover public daily data ingestion and data-audit outputs. Phase 1 does not compute realised variance, implied variance, VRP, forecasts, regimes, strategies, backtests, or broker signals.

## Commit Policy

Commit:

```text
configs/data_sources.yaml
configs/markets.yaml
scripts/download_data.py
src/vrp/data/
tests/test_data_loaders.py
tests/test_data_schema.py
tests/fixtures/
docs/phases/phase_01_data_ingestion.md
src/vrp/data/README.md
```

Do not commit by default:

```text
data/raw/*.parquet
data/processed/*.parquet
data/manual/**/*.csv
reports/tables/data_audit.csv
```

## Expected Local Raw Outputs

```text
data/raw/us_vix_cboe.parquet
data/raw/us_vix_fred.parquet
data/raw/us_vix_yahoo.parquet
data/raw/us_spx_yahoo.parquet
data/raw/us_spy_yahoo.parquet
data/raw/india_vix_yahoo.parquet
data/raw/india_nifty_yahoo.parquet
```

Optional manual-source outputs:

```text
data/raw/india_vix_nse.parquet
data/raw/india_nifty_nse.parquet
```

## Expected Local Processed Outputs

```text
data/processed/us_vix.parquet
data/processed/us_underlying.parquet
data/processed/india_vix.parquet
data/processed/india_underlying.parquet
```

## Expected Local Report Outputs

```text
reports/tables/data_audit.csv
```

Audit table columns:

```text
market
dataset
source
symbol
start_date
end_date
n_rows
n_missing_close
n_duplicate_dates
min_close
max_close
validation_status
```

## Validation Commands

Dry run:

```bash
python scripts/download_data.py --market ALL --source all --dry-run
```

US public sources:

```bash
python scripts/download_data.py --market US --source all --force
```

India Yahoo fallback:

```bash
python scripts/download_data.py --market INDIA --source yahoo --force
```

Manual NSE override:

```bash
python scripts/download_data.py --market INDIA --source nse --source-id nse_india_vix --local-csv data/manual/nse/india_vix.csv --force
```

Tests:

```bash
pytest tests/test_data_loaders.py tests/test_data_schema.py
```

## Review Substitute

Send these instead of committing generated data:

```text
download dry-run output
pytest output
reports/tables/data_audit.csv preview
data/raw file tree
data/processed file tree
head/tail of processed parquet files printed in terminal
```

## Reviewer Notes

Phase 1 review should verify canonical OHLCV schema, source priority, manual override behavior, data-audit reporting, and local-only handling of all downloaded or processed data.
