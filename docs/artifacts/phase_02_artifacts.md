# Phase 2 Artifacts — Realised Variance

## Policy

Phase 2 generated files are local-only by default.

Commit:

```text
source code
tests
docs
README files
.gitkeep placeholders
```

Do not commit:

```text
full generated parquet panels
diagnostic CSV files
diagnostic JSON files
generated figures
logs
```

Generated artifacts should be regenerated locally from the documented producer commands.

## Producer Command

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute |
|---|---|---|---:|---|---|---|
| US realised variance panel | `data/processed/us_rv.parquet` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No | Generated processed panel | `date`, `market`, `symbol`, returns, daily RV columns, 22d annualised RV columns | `rv_summary.csv` preview, `head()`, `tail()`, tests |
| India realised variance panel | `data/processed/india_rv.parquet` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No | Generated processed panel | `date`, `market`, `symbol`, returns, daily RV columns, 22d annualised RV columns | `rv_summary.csv` preview, `head()`, `tail()`, tests |
| RV summary table | `reports/tables/rv_summary.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic table | `market`, `symbol`, `column`, `mean`, `median`, `std`, `min`, `max`, `p95`, `count`, `missing` | CSV preview |
| RV estimator correlations | `reports/tables/rv_estimator_correlations.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic table | `market`, `symbol`, estimator pair, `correlation` | CSV preview |
| RV metadata | `reports/tables/rv_metadata.json` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated run metadata | phase, inputs, primary estimator, rolling convention, annualisation | JSON preview |
| US RV estimator figure | `reports/figures/rv_estimators_us.png` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic figure | Annualised RV estimator comparison | Screenshot if needed |
| India RV estimator figure | `reports/figures/rv_estimators_india.png` | `python scripts/build_features.py --market ALL --feature rv --window 22` | No by default | Generated diagnostic figure | Annualised RV estimator comparison | Screenshot if needed |

## RV Panel Key Columns

```text
date
market
symbol
log_return
simple_return
gap_return
intraday_return
rv_cc_daily
rv_parkinson_daily
rv_gk_daily
rv_rs_daily
rv_cc_22d_ann
rv_parkinson_22d_ann
rv_gk_22d_ann
rv_rs_22d_ann
rv_yz_22d_ann
```

## Primary RV Column

```text
rv_gk_22d_ann
```

Garman-Klass is the primary realised variance estimator for the main research path.

## Robustness RV Columns

```text
rv_cc_22d_ann
rv_parkinson_22d_ann
rv_rs_22d_ann
rv_yz_22d_ann
```

These support estimator-sensitivity checks.

## Forbidden / Intentionally Absent Columns

```text
rv_yz_daily
```

Yang-Zhang is rolling-window only in this project. Do not create or document a daily Yang-Zhang estimator unless Phase 2 is explicitly re-scoped.

## Annualisation Convention

```text
rv_*_22d_ann = 252 * trailing_22_day_mean(daily_variance)
```

The output is annualised variance, not annualised volatility.

## No-Lookahead Notes

- Rolling RV uses trailing windows only.
- No centered rolling window is allowed.
- Phase 2 does not create forward, future, ex-post, or label columns.
- Phase 3 is responsible for explicit lagging before VRP feature use.

## Sensitivity and Reproducibility Notes

- RV panels depend on Phase 1 cleaned OHLC files.
- Range-based estimators are sensitive to bad high/low values.
- Generated artifacts should be regenerated locally rather than committed.
- Review should use tests, metadata, summaries, and small previews instead of full parquet files.

## Lightweight Review Packet

Use these instead of sending large generated files:

```text
terminal output:
python scripts/build_features.py --market ALL --feature rv --window 22

terminal output:
pytest tests/test_rv_estimators.py

preview:
reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json
```

Do not send full parquet files unless explicitly requested.
