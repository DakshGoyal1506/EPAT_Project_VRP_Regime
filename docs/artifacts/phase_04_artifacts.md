# Phase 4 Artifacts - HAR-RV Forecasting and HAR-Based VRP

Generated Phase 4 artifacts are local-only by default.

## Commit Policy

Commit:

```text
code
configs
tests
documentation
README files
.gitkeep placeholders
```

Do not commit by default:

```text
data/processed/*_har_forecast.parquet
data/processed/*_vrp_har.parquet
reports/tables/har_*.csv
reports/tables/har_*.json
reports/figures/har_*.png
```

Small summary tables or final-report figures may be committed later only if explicitly approved.

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute |
|---|---|---|---:|---|---|---|
| US HAR forecast panel | `data/processed/us_har_forecast.parquet` | `python scripts/train_har.py --market US --mode expanding --force` | No | Generated processed panel | `date`, `market`, `target_end_date`, `rv_gk_22d_forward_ann_label`, `har_rv_d_lag1_ann`, `har_rv_w_lag1_ann`, `har_rv_m_lag1_ann`, `har_rv_gk_22d_forecast_ann`, `har_forecast_available`, `har_blocked_reason` | head/tail preview and schema printout |
| India HAR forecast panel | `data/processed/india_har_forecast.parquet` | `python scripts/train_har.py --market INDIA --mode expanding --force` | No | Generated processed panel | Same as US HAR forecast panel | head/tail preview and schema printout |
| US HAR-VRP panel | `data/processed/us_vrp_har.parquet` | `python scripts/train_har.py --market US --mode expanding --force` | No | Generated processed panel | all Phase 3 VRP columns plus `har_rv_gk_22d_forecast_ann`, `har_forecast_available`, `har_blocked_reason`, `vrp_har_gk`, `vrp_har_gk_positive` | row-count check and unavailable-VRP validation |
| India HAR-VRP panel | `data/processed/india_vrp_har.parquet` | `python scripts/train_har.py --market INDIA --mode expanding --force` | No | Generated processed panel | same as US HAR-VRP panel | row-count check and unavailable-VRP validation |
| Forecast accuracy table | `reports/tables/har_forecast_accuracy.csv` | `python scripts/train_har.py --market ALL --mode expanding --force --write-reports` | No by default | Generated diagnostic table | `market`, `forecast_col`, `target_col`, `n_obs`, `mse`, `rmse`, `mae`, `qlike`, `bias`, `correlation` | CSV preview |
| Coefficient history | `reports/tables/har_coefficients.csv` | same as above | No by default | Generated diagnostic table | `date`, `market`, `coef_const`, `coef_har_rv_d_lag1_ann`, `coef_har_rv_w_lag1_ann`, `coef_har_rv_m_lag1_ann`, `hac_available` | CSV preview |
| HAR-VRP summary | `reports/tables/har_vrp_summary.csv` | same as above | No by default | Generated diagnostic table | market-level descriptive statistics for IV, backward VRP, HAR forecast, and HAR-VRP | CSV preview |
| HAR metadata | `reports/tables/har_metadata.json` | same as above | No by default | Generated run metadata | model type, target definition, timing rule, feature columns, output columns, backend settings | JSON preview |
| No-lookahead audit | `reports/tables/har_no_lookahead_audit.csv` | same as above | No by default | Critical audit output | `market`, `forecast_date`, `max_training_target_end_date`, `rule_target_end_before_forecast_date`, `forecast_available`, `blocked_reason` | audit validation output |
| US forecast figure | `reports/figures/har_forecast_us.png` | same as above | No by default | Generated figure | realised forward RV vs HAR forecast | screenshot if needed |
| India forecast figure | `reports/figures/har_forecast_india.png` | same as above | No by default | Generated figure | realised forward RV vs HAR forecast | screenshot if needed |
| US residual figure | `reports/figures/har_residuals_us.png` | same as above | No by default | Generated figure | forecast residuals | screenshot if needed |
| India residual figure | `reports/figures/har_residuals_india.png` | same as above | No by default | Generated figure | forecast residuals | screenshot if needed |
| US HAR-VRP figure | `reports/figures/har_vrp_us.png` | same as above | No by default | Generated figure | backward VRP vs HAR-VRP | screenshot if needed |
| India HAR-VRP figure | `reports/figures/har_vrp_india.png` | same as above | No by default | Generated figure | backward VRP vs HAR-VRP | screenshot if needed |

## Reproducibility Notes

Primary full-market command:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend torch_batched --torch-device cuda --torch-dtype float64 --coefficient-hac-frequency none
```

CPU fallback:

```bash
python scripts/train_har.py --market ALL --mode expanding --force --backend cpu_numpy_batched --coefficient-hac-frequency none
```

## Required Review Substitutes

Instead of committing generated parquet or report outputs, provide:

```text
pytest output
train_har.py console output
har_no_lookahead_audit.csv preview
har_forecast_accuracy.csv preview
har_coefficients.csv preview
schema/head/tail of *_har_forecast.parquet
schema/head/tail of *_vrp_har.parquet
HAR-VRP unavailable-row validation output
```

## Sensitivity

Phase 4 artifacts are not broker-sensitive, but they are generated research outputs. Keep them local unless explicitly selected as small final-report artifacts.
