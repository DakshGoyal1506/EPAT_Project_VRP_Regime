# Data

This directory stores local generated data artifacts.

Most files under this directory are intentionally not tracked by Git.

## Directory Contract

| Path | Purpose | Commit? |
|---|---|---:|
| `data/raw/` | Source-specific downloaded or ingested files | No |
| `data/interim/` | Temporary cleaned/intermediate files | No |
| `data/processed/` | Canonical model-ready panels | No |
| `data/manual/` | Manual CBOE/NSE CSV downloads used as local overrides | No |
| `data/manual/cboe/` | Optional manually downloaded CBOE files | No |
| `data/manual/nse/` | Optional manually downloaded NSE files | No |
| `data/broker_cache/` | Optional broker/paper-signal cache | No |

Only README files and `.gitkeep` placeholders should be tracked here.

## Phase 1 Canonical Processed Outputs

Generated locally by ingestion:

```text
data/processed/us_vix.parquet
data/processed/us_underlying.parquet
data/processed/india_vix.parquet
data/processed/india_underlying.parquet
```

## Later Canonical Outputs

Generated locally by later phases:

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet
data/processed/us_iv.parquet
data/processed/india_iv.parquet
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
data/processed/us_har_forecast.parquet
data/processed/india_har_forecast.parquet
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet
```

## Rules

1. Do not commit raw data.
2. Do not commit processed parquet panels.
3. Do not commit manual downloaded CSVs.
4. Do not commit broker cache.
5. Regenerate data using scripts and configs.
6. Use `docs/artifact_inventory.md` to document generated outputs and review substitutes.
