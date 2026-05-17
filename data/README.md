# Data

This directory stores the phase inputs, intermediates, and canonical processed outputs for the project.

## Directory Contract

- `data/raw/` - source-specific raw files produced by ingestion.
- `data/interim/` - temporary or cleaned intermediates that are not final outputs.
- `data/processed/` - canonical datasets consumed by the feature pipeline and notebooks.
- `data/manual/` - optional manually downloaded source files used by ingestion overrides.
- `data/broker_cache/` - optional broker-side cache files, not intended for version control.

## Canonical Phase 1 Inputs

These are the key files produced by the ingestion layer and consumed by later phases:

- `data/processed/us_vix.parquet`
- `data/processed/us_underlying.parquet`
- `data/processed/india_vix.parquet`
- `data/processed/india_underlying.parquet`

## Canonical Phase 2 and 3 Outputs

Later phases write these canonical outputs here:

- `data/processed/us_rv.parquet`
- `data/processed/india_rv.parquet`
- `data/processed/us_iv.parquet`
- `data/processed/india_iv.parquet`
- `data/processed/us_vrp.parquet`
- `data/processed/india_vrp.parquet`

## Rules

- Do not commit large raw, interim, or broker cache files.
- Keep the processed parquet files stable and reproducible.
- Treat the `processed/` directory as the handoff point between ingestion, feature building, notebooks, and reporting.

## Related Docs

- [scripts/README.md](../scripts/README.md)
- [src/vrp/features/README.md](../src/vrp/features/README.md)
- [notebooks/README.md](../notebooks/README.md)
