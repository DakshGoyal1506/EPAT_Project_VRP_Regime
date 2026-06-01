# Reports

This package provides reporting and diagnostic utilities that assemble reproducible tables and figures from canonical Phase 2/3/4 inputs. Keep reporting focused on presentation, audit checks, and stable output contracts; heavy validation belongs in `src/vrp/features/` and forecasting.

Responsibilities

- Summarise VRP and RV panels into stable tables for manuscripts and appendices.
- Emit diagnostic audit tables that verify no-lookahead conditions and data availability.
- Produce publication-ready figures and CSV tables under `reports/figures` and `reports/tables`.

Key modules

- `rv_diagnostics.py` — realised-variance diagnostics and calendar mismatch reports.
- `vrp_diagnostics.py` — VRP summaries, metadata writers, and plotting helpers.

Common outputs

- Summary tables: `reports/tables/vrp_summary.csv`, `reports/tables/vrp_metadata.json`, `reports/tables/calendar_mismatches.csv`
- HAR diagnostics (if HAR forecasts are available): `reports/tables/har_forecast_accuracy.csv`, `reports/tables/har_coefficients.csv`, `reports/tables/har_vrp_summary.csv`, `reports/tables/har_no_lookahead_audit.csv`
- Figures: market-specific IV/RV/VRP plots in `reports/figures/`

Audit contract

Every no-lookahead audit row must satisfy:

```text
max_training_target_end_date < forecast_date
```

If this condition fails, the audit helper flags the offending rows and the reporting pipeline stops.

Examples (commands)

```bash
# Build VRP tables from processed features
python scripts/build_features.py --market ALL --feature vrp

# Produce backtest diagnostics (requires processed backtest outputs)
python scripts/generate_backtest_diagnostics.py --out reports/tables/backtest_diagnostics.csv
```

Notes

- Reporting functions are defensive: missing columns are skipped and non-numeric data is coerced to `NaN` for reporting only.
- Metadata files describe phase, primary estimator, robustness estimator list, and horizon settings.
