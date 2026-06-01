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
  conda activate epat
  ```

- Install package in editable/development mode (one-time):

  ```bash
  pip install -e .
  pip install -e ".[dev]"
  ```

- Dry-run ingestion (no downloads, no writes):

  ```bash
  python scripts/download_data.py --dry-run
  ```

- Download only FRED sources for US and overwrite outputs:

  ```bash
  python scripts/download_data.py --market US --source fred --force
  ```

- Download only CBOE sources for US (use `--source-id` for a specific source):

  ```bash
  python scripts/download_data.py --market US --source cboe --force
  python scripts/download_data.py --market US --source cboe --source-id cboe_vix --force
  ```

- Download Yahoo sources for US:

  ```bash
  python scripts/download_data.py --market US --source yahoo --force
  ```

- Use a local CSV override (manual CBOE/NSE downloads):

  ```bash
  python scripts/download_data.py --market US --source cboe --source-id cboe_vix --local-csv data/manual/cboe/VIX_History.csv --force
  ```

- Run tests relevant to data ingestion and validators:

  ```bash
  pytest tests/test_data_loaders.py tests/test_data_schema.py -q
  ```

Outputs produced by the ingestion step:

- Raw per-source Parquet files: `data/raw/*.parquet`
- Processed canonical datasets (one per processed_outputs entry): `data/processed/*.parquet`
- Data audit CSV: `reports/tables/data_audit.csv`

If you change loader logic, add unit tests under `tests/` and run the test suite before committing.
