# Phase 1 — Data Ingestion

## Status

Complete / frozen.

Phase 1 implements public daily market data ingestion. It does not compute realised variance, implied variance, VRP, forecasts, regimes, strategies, or backtests.

## Objective

Implement reliable loaders for public-data research inputs across US and India.

Core series:

```text
US:
- CBOE VIX daily historical data
- FRED VIXCLS backup
- Yahoo Finance ^GSPC / SPY OHLCV
- Yahoo Finance ^VIX fallback

India:
- Yahoo Finance ^NSEI
- Yahoo Finance ^INDIAVIX
- NSE India VIX manual CSV path
- NSE NIFTY manual CSV path
```

## Phase Boundary

Included:

```text
source configs
market configs
source-specific loaders
manual CSV ingestion
canonical OHLCV schema
validation
Parquet IO
raw source cache
processed canonical datasets
data audit table
loader tests with mocks/fixtures
```

Excluded:

```text
realised variance
implied variance
VRP
HAR-RV forecasting
regime modelling
strategy signals
backtests
broker data
calendar merging between US and India
```

## Main Files

```text
configs/data_sources.yaml
configs/markets.yaml

src/vrp/data/base.py
src/vrp/data/schema.py
src/vrp/data/validators.py
src/vrp/data/cleaners.py
src/vrp/data/io.py
src/vrp/data/yahoo_loader.py
src/vrp/data/fred_loader.py
src/vrp/data/cboe_loader.py
src/vrp/data/nse_loader.py

scripts/download_data.py

tests/test_data_schema.py
tests/test_data_loaders.py
tests/fixtures/
notebooks/01_data_audit.ipynb
```

## Canonical Schema

```text
date
open
high
low
close
adj_close
volume
source
market
symbol
```

## Loader Design

```text
scripts/download_data.py
    ↓
config selection
    ↓
source-specific loader
    ↓
cleaner
    ↓
validator
    ↓
raw source parquet
    ↓
explicit processed-source priority
    ↓
processed canonical parquet
    ↓
data audit table
```

## Source Priority

US VIX:

```text
1. CBOE VIX
2. FRED VIXCLS
3. Yahoo ^VIX
```

US underlying:

```text
1. Yahoo ^GSPC
2. Yahoo SPY
```

India VIX:

```text
1. NSE India VIX manual CSV
2. Yahoo ^INDIAVIX
```

India underlying:

```text
1. Yahoo ^NSEI
2. NSE NIFTY manual CSV if enabled
```

## Commands

Dry run:

```bash
python scripts/download_data.py --market ALL --source all --dry-run
```

US all sources:

```bash
python scripts/download_data.py --market US --source all --force
```

US Yahoo only:

```bash
python scripts/download_data.py --market US --source yahoo --force
```

US CBOE only:

```bash
python scripts/download_data.py --market US --source cboe --force
```

US FRED only:

```bash
python scripts/download_data.py --market US --source fred --force
```

India Yahoo fallback:

```bash
python scripts/download_data.py --market INDIA --source yahoo --force
```

India NSE manual override:

```bash
python scripts/download_data.py --market INDIA --source nse --source-id nse_india_vix --local-csv data/manual/nse/india_vix.csv --force
```

Tests:

```bash
pytest tests/test_data_loaders.py tests/test_data_schema.py
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

Optional manual outputs:

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

Audit columns:

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

## Commit Policy

Commit:

```text
configs
source code
script
tests
fixtures
docs
README files
.gitkeep placeholders
```

Do not commit:

```text
data/raw/*.parquet
data/processed/*.parquet
data/manual/**/*.csv
reports/tables/data_audit.csv unless explicitly approved
```

## Known Phase 1 Limitations

1. Yahoo Finance can fail or change response format.
2. NSE scripted access can be blocked; manual CSV override is preferred.
3. CBOE VIX is treated as close-only for implied-volatility research use.
4. FRED VIXCLS is close-only and mapped to OHLCV shape explicitly.
5. US and India calendars are not merged in Phase 1.
6. No price forward-fill is allowed.

## Validation Checklist

Phase 1 is closed when:

```text
loader tests pass
schema tests pass
dry-run works
Yahoo US works or fails clearly
Yahoo India works or fails clearly
CBOE works or gives manual instruction
FRED works or fails clearly
processed files follow canonical schema
data audit table is produced
no generated data is tracked
```

## Review Packet

Send:

```text
pytest output
download dry-run output
data/raw file tree
data/processed file tree
reports/tables/data_audit.csv preview
head/tail of processed parquet files
scripts/download_data.py
src/vrp/data/*.py
tests/test_data_loaders.py
tests/test_data_schema.py
```
