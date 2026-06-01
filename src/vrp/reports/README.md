# Reports

This package contains report assembly helpers used to build tables, figures, and summary outputs from reproducible inputs.

It should stay focused on presentation logic, not on source-data ingestion or model training.
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

Phase 4 HAR diagnostics

The reporting layer also supports Phase 4 HAR diagnostics and tables produced
by the HAR forecasting engine. Key outputs:

- `reports/tables/har_forecast_accuracy.csv`
- `reports/tables/har_coefficients.csv`
- `reports/tables/har_vrp_summary.csv`
- `reports/tables/har_no_lookahead_audit.csv`

These are written by the `write_har_diagnostics` helper and are intended for
final inspection and inclusion in reports.

Phase 4 HAR-RV Reports

Phase 4 writes the following reproducible report artifacts:

```text
reports/tables/har_forecast_accuracy.csv
reports/tables/har_coefficients.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_metadata.json
reports/tables/har_no_lookahead_audit.csv
reports/figures/har_forecast_us.png
reports/figures/har_forecast_india.png
reports/figures/har_residuals_us.png
reports/figures/har_residuals_india.png
reports/figures/har_vrp_us.png
reports/figures/har_vrp_india.png
```

Key audit condition:

```text
max_training_target_end_date < forecast_date
```

This must hold for every available HAR forecast row.

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
