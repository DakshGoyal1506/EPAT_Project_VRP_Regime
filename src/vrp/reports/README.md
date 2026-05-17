# VRP Reports

This package contains the reporting and diagnostic utilities used after feature construction. It does not build signals, regimes, forecasts, or backtests.

## Responsibilities

- summarize RV and VRP panels
- capture Phase 3 metadata
- write stable tables and figures under `reports/`
- keep reporting separate from feature validation and live feature construction

## Modules

### `rv_diagnostics.py`

Diagnostics for realised variance panels.

### `vrp_diagnostics.py`

Diagnostics for VRP panels.

- builds descriptive summaries for primary and robustness columns
- writes metadata JSON describing the construction rules
- plots the primary IV / RV / VRP series for each market

## Outputs

The VRP reporting path writes:

- `reports/tables/vrp_summary.csv`
- `reports/tables/vrp_metadata.json`
- `reports/tables/calendar_mismatches.csv`
- `reports/figures/us_iv_rv_vrp.png`
- `reports/figures/india_iv_rv_vrp.png`

## Behavior

The summary layer is intentionally permissive:

- missing columns are skipped
- values are coerced to numeric for reporting
- non-numeric values become `NaN` in the report layer only

Validation should happen earlier in `src/vrp/features/`.

## Metadata

The metadata file records:

- phase name
- primary estimator
- robustness estimator list
- formulas and horizon settings
- feature registry metadata

## Plotting Contract

The main VRP figure stays focused on the primary GK path:

- `iv_ann`
- `rv_gk_22d_ann_lag1`
- `vrp_backward_gk`
- `vrp_forward_expost_gk_label`

Robustness estimators are summarized in tables rather than crowded into the figure.

## Common Usage

```bash
python scripts/build_features.py --market ALL --feature vrp
```
