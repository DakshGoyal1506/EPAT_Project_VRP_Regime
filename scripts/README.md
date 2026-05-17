# Scripts

This folder contains the command-line entry points for the EPAT VRP project.

The scripts are meant to be run from the repository root and are thin orchestration layers around the reusable code in `src/vrp/`.

## Script Inventory

### `build_features.py`

Main feature pipeline driver.

It can build:

- realised variance panels
- implied variance panels
- full VRP panels and associated diagnostics

This is the most important script in the repository for Phase 2 and Phase 3 work.

### `download_data.py`

Downloads or refreshes source data used by the project.

### `run_backtest.py`

Runs the vectorised backtest layer for strategy evaluation.

### `run_ibkr_paper_signal.py`

Runs the optional paper-signal integration layer.

### `run_robustness.py`

Runs robustness checks for the research pipeline.

### `train_regimes.py`

Runs regime-model training workflows.

## `build_features.py`

This script is the central Phase 2 / Phase 3 build entry point.

It reads raw or processed inputs, constructs feature panels, saves parquet outputs, and optionally writes diagnostics.

### Supported Features

Use the `--feature` flag to select the build target:

- `rv` builds realised variance panels
- `iv` builds implied variance panels
- `vrp` builds implied variance plus VRP panels

### Supported Markets

Use the `--market` flag to choose the market scope:

- `US`
- `INDIA`
- `ALL`

### Core Arguments

- `--market`: market to process
- `--feature`: `rv`, `iv`, or `vrp`
- `--window`: trailing RV window length, default `22`
- `--horizon`: forward ex-post label horizon, default `22`
- `--annualization-periods`: annualization factor, default `252`
- `--max-vix-value`: upper bound check for VIX-style close values, default `200`
- `--skip-diagnostics`: write parquet outputs but skip reports and figures

### Example Commands

Build realised variance for both markets:

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

Build implied variance for both markets:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build full VRP outputs and diagnostics:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

Build VRP outputs without diagnostics:

```bash
python scripts/build_features.py --market ALL --feature vrp --skip-diagnostics
```

### Expected VRP Outputs

When the VRP build runs successfully, the primary outputs are:

- `data/processed/us_iv.parquet`
- `data/processed/india_iv.parquet`
- `data/processed/us_vrp.parquet`
- `data/processed/india_vrp.parquet`
- `reports/tables/vrp_summary.csv`
- `reports/tables/vrp_metadata.json`
- `reports/tables/calendar_mismatches.csv`
- `reports/figures/us_iv_rv_vrp.png`
- `reports/figures/india_iv_rv_vrp.png`

### VRP Contract

The VRP build script preserves the primary live feature contract:

- `iv_ann`
- `rv_gk_22d_ann_lag1`
- `vrp_backward_gk`
- `vrp_backward_gk_positive`
- `rv_gk_22d_forward_ann_label`
- `vrp_forward_expost_gk_label`

It may also surface robustness-only backward VRP diagnostics for other estimators, but those columns are not part of the primary live feature registry.

### Validation

After editing the build workflow, run:

```bash
pytest
```

If you are specifically changing the VRP build path, a targeted run is also useful:

```bash
pytest tests/test_vrp_alignment.py tests/test_no_lookahead.py
```
