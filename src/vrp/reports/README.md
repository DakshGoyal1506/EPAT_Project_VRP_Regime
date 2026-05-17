# VRP Reports

This package contains the reporting and diagnostic utilities used after feature construction.

The modules here do not build signals, regimes, forecasts, or backtests. Their role is to summarize what the feature pipeline produced and to write stable tables and figures for inspection.

## Purpose

The reports package is responsible for:

- descriptive statistics for VRP panels
- metadata capture for the Phase 3 feature pipeline
- plot generation for IV, lagged RV, and VRP series
- simple report file writing under `reports/`

## Modules

### `rv_diagnostics.py`

Diagnostics for realised variance panels.

Typical responsibilities include:

- summary tables for RV columns
- panel-level diagnostics
- figures for RV inspection

### `vrp_diagnostics.py`

Diagnostics for VRP panels.

Responsibilities:

- build a descriptive summary table for primary and robustness VRP columns
- write metadata JSON describing the construction rules
- plot the primary IV/RV/VRP series for each market
- keep the reporting layer separate from live feature construction

## VRP Diagnostics Outputs

The VRP reporting path writes the following artifacts by default:

- `reports/tables/vrp_summary.csv`
- `reports/tables/vrp_metadata.json`
- `reports/figures/us_iv_rv_vrp.png`
- `reports/figures/india_iv_rv_vrp.png`

Calendar mismatch reports are written separately by the build script:

- `reports/tables/calendar_mismatches.csv`

## Summary Table Behavior

The VRP summary table is intentionally permissive:

- missing columns are skipped
- values are coerced to numeric for reporting
- non-numeric values become `NaN` in the report layer only

This means the reporting package is not a validator. Validation should happen earlier in `src/vrp/features/`.

## Metadata Contract

The VRP metadata file records:

- phase name
- primary estimator
- robustness estimator list
- formula definitions
- horizon and annualization settings
- feature registry metadata

This metadata is intended to make the Phase 3 output self-describing and reproducible.

## Plotting Contract

The main VRP plot is intentionally focused on the primary GK path:

- `iv_ann`
- `rv_gk_22d_ann_lag1`
- `vrp_backward_gk`
- `vrp_forward_expost_gk_label`

Robustness estimators are summarized in tables rather than crowded into the figure.

## Common Usage

From the repository root, the typical entry point is the build script:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

That command builds the panels and writes the report outputs above.
