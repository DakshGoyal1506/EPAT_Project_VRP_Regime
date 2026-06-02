# Report Tables

Generated diagnostic and summary tables are written here.

## Commit Policy

By default, generated tables stay local.

Commit only:

```text
README.md
.gitkeep
small final-report-ready tables explicitly approved for review
```

Do not commit by default:

```text
*.csv
*.json
*.parquet
*.xlsx
```

## Examples

Local generated tables may include:

```text
data_audit.csv
vrp_summary.csv
har_forecast_accuracy.csv
har_coefficients.csv
har_vrp_summary.csv
har_metadata.json
har_no_lookahead_audit.csv
backtest_summary.csv
phase_10/*.csv
phase_11/*.csv
```

Use `docs/artifact_inventory.md` to document which local table should be sent as a review substitute.
