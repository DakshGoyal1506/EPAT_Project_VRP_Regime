# Data utilities and ingestion (src/vrp/data)

This folder contains data ingestion helpers, canonical schema definitions, cleaners,
and source-specific loaders used by Phase 1 of the EPAT VRP project.

Phase status, artifact policy, and reproducibility notes are tracked in `docs/`.

Key scripts and modules used by this package live in the repository `scripts/` and
`src/vrp/data/` respectively. The command-line entrypoint for Phase 1 ingestion is
`scripts/download_data.py` (dry-run safe by default).

Common commands (run from repository root):

- Activate the development environment (Conda):

  ```powershell
  # `src/vrp/data`

  Data ingestion, schema, cleaning, validation, and IO utilities.

  ## Phase Ownership

  Primary phase:

  ```text
  Phase 1 — Public data ingestion
  ```

  ## Responsibilities

  This package handles:

  ```text
  canonical OHLCV schema
  source-specific loaders
  manual CSV ingestion
  column standardisation
  data validation
  Parquet IO
  data audit rows
  ```

  ## Key Modules

  | Module            | Purpose                                                                  |
  | ----------------- | ------------------------------------------------------------------------ |
  | `schema.py`       | Canonical OHLCV and audit schemas                                        |
  | `validators.py`   | Strict schema, missingness, duplicate-date, sorted-date, OHLC validation |
  | `cleaners.py`     | Source-column normalisation into canonical OHLCV                         |
  | `io.py`           | Parquet save/load helpers                                                |
  | `base.py`         | Loader base types and ingestion result container                         |
  | `yahoo_loader.py` | Yahoo Finance OHLCV ingestion                                            |
  | `fred_loader.py`  | FRED close-series ingestion                                              |
  | `cboe_loader.py`  | CBOE VIX ingestion; close-only VIX canonicalisation                      |
  | `nse_loader.py`   | Manual/local NSE CSV ingestion                                           |

  ## Canonical OHLCV Columns

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

  ## Rules

  1. Do not forward-fill prices silently.
  2. Do not silently merge sources.
  3. Do not merge US and India calendars in this package.
  4. Do not calculate realised variance here.
  5. Do not calculate VRP here.
  6. Do not train models here.
  7. Loader tests must not depend on live internet.
  8. Local/manual CSVs stay under `data/manual/` and are not committed.
Outputs produced by the ingestion step:

- Raw per-source Parquet files: `data/raw/*.parquet`
- Processed canonical datasets (one per processed_outputs entry): `data/processed/*.parquet`
- Data audit CSV: `reports/tables/data_audit.csv`

If you change loader logic, add unit tests under `tests/` and run the test suite before committing.
