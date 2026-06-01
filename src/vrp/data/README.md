# `src/vrp/data`

Data ingestion, schema, cleaning, validation, and IO utilities.

## Phase Ownership

Primary phase:

```text
Phase 1 — Public data ingestion
```

## Purpose

This package turns source-specific public market data into the project’s canonical daily OHLCV schema.

It handles source loading, local/manual CSV ingestion, column standardisation, validation, Parquet IO, and audit-row construction.

It does not compute realised variance, implied variance, VRP, forecasts, regimes, strategy signals, backtests, or broker outputs.

## Module Map

| Module            | Purpose                                                                           |
| ----------------- | --------------------------------------------------------------------------------- |
| `schema.py`       | Canonical OHLCV and audit schemas                                                 |
| `validators.py`   | Strict schema, numeric, missingness, duplicate-date, sorted-date, OHLC validation |
| `cleaners.py`     | Source-column normalisation into canonical OHLCV                                  |
| `io.py`           | Parquet save/load helpers                                                         |
| `base.py`         | Loader base types and ingestion result container                                  |
| `yahoo_loader.py` | Yahoo Finance OHLCV ingestion                                                     |
| `fred_loader.py`  | FRED close-series ingestion                                                       |
| `cboe_loader.py`  | CBOE VIX ingestion; close-only VIX canonicalisation                               |
| `nse_loader.py`   | Manual/local NSE CSV ingestion                                                    |

## Canonical OHLCV Schema

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

## Loader Architecture

```text
scripts/download_data.py
  ↓
configs/data_sources.yaml
  ↓
source-specific loader
  ↓
cleaner
  ↓
validator
  ↓
raw Parquet
  ↓
processed Parquet selected by explicit source priority
  ↓
data audit table
```

## Supported Source Families

| Source family | Loader            | Notes                                                  |
| ------------- | ----------------- | ------------------------------------------------------ |
| Yahoo Finance | `yahoo_loader.py` | Used for SPX, SPY, VIX, NIFTY, India VIX               |
| FRED          | `fred_loader.py`  | Used for VIXCLS close-only backup                      |
| CBOE          | `cboe_loader.py`  | Used for official VIX; close-only canonicalisation     |
| NSE           | `nse_loader.py`   | Manual/local CSV first; no fragile scraping dependency |

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

## Expected Local Outputs

Raw source-specific files:

```text
data/raw/us_vix_cboe.parquet
data/raw/us_vix_fred.parquet
data/raw/us_vix_yahoo.parquet
data/raw/us_spx_yahoo.parquet
data/raw/us_spy_yahoo.parquet
data/raw/india_vix_yahoo.parquet
data/raw/india_nifty_yahoo.parquet
```

Processed canonical files:

```text
data/processed/us_vix.parquet
data/processed/us_underlying.parquet
data/processed/india_vix.parquet
data/processed/india_underlying.parquet
```

Audit table:

```text
reports/tables/data_audit.csv
```

## Data-Layer Rules

1. Do not forward-fill prices silently.
2. Do not silently merge sources.
3. Do not merge US and India calendars here.
4. Do not calculate realised variance here.
5. Do not calculate implied variance or VRP here.
6. Do not train models here.
7. Do not create strategy signals here.
8. Do not access broker data here.
9. Loader tests must not depend on live internet.
10. Manual CSV files stay under `data/manual/` and are not committed.

