# Phase 3 Artifacts — Implied Variance and VRP

## Policy

Phase 3 generated files are local-only by default.

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
full IV panels
full VRP panels
diagnostic CSV files
diagnostic JSON files
generated figures
logs
```

Generated artifacts should be regenerated locally from the documented producer commands.

## Producer Commands

Build IV panels only:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build IV, VRP panels, and VRP diagnostics:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute |
|---|---|---|---:|---|---|---|
| US implied variance panel | `data/processed/us_iv.parquet` | `python scripts/build_features.py --market ALL --feature iv` or `python scripts/build_features.py --market ALL --feature vrp` | No | Generated processed panel | `date`, `market`, `iv_symbol`, `iv_close`, `iv_ann` | `head()`, `tail()`, formula check |
| India implied variance panel | `data/processed/india_iv.parquet` | `python scripts/build_features.py --market ALL --feature iv` or `python scripts/build_features.py --market ALL --feature vrp` | No | Generated processed panel | `date`, `market`, `iv_symbol`, `iv_close`, `iv_ann` | `head()`, `tail()`, formula check |
| US VRP panel | `data/processed/us_vrp.parquet` | `python scripts/build_features.py --market ALL --feature vrp` | No | Generated processed panel | IV columns, lagged GK RV, backward GK VRP, forward ex-post labels, robustness columns | `vrp_summary.csv` preview, metadata, tests |
| India VRP panel | `data/processed/india_vrp.parquet` | `python scripts/build_features.py --market ALL --feature vrp` | No | Generated processed panel | IV columns, lagged GK RV, backward GK VRP, forward ex-post labels, robustness columns | `vrp_summary.csv` preview, metadata, tests |
| VRP summary | `reports/tables/vrp_summary.csv` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated diagnostic table | `market`, `column`, `mean`, `median`, `std`, `min`, `max`, `p05`, `p95`, `count`, `missing`, `positive_count`, `positive_ratio` | CSV preview |
| VRP metadata | `reports/tables/vrp_metadata.json` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated run metadata | formulas, primary estimator, robustness estimators, feature registry, no-forward-fill policy | JSON preview |
| Calendar mismatch report | `reports/tables/calendar_mismatches.csv` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated alignment diagnostic | `market`, date ranges, row counts, common/IV-only/RV-only dates | CSV preview |
| US IV/RV/VRP figure | `reports/figures/us_iv_rv_vrp.png` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated diagnostic figure | IV vs lagged RV, backward VRP, forward label | Screenshot if needed |
| India IV/RV/VRP figure | `reports/figures/india_iv_rv_vrp.png` | `python scripts/build_features.py --market ALL --feature vrp` | No by default | Generated diagnostic figure | IV vs lagged RV, backward VRP, forward label | Screenshot if needed |

## IV Panel Schema

```text
date
market
iv_symbol
iv_close
iv_ann
```

Formula:

```text
iv_ann = (iv_close / 100) ** 2
```

Column meanings:

```text
iv_close = VIX / India VIX close level
iv_ann = annualised implied variance proxy
```

## VRP Panel Key Columns

```text
date
market
underlying_symbol
iv_symbol
iv_close
iv_ann
rv_gk_daily
rv_gk_22d_ann
rv_gk_22d_ann_lag1
vrp_backward_gk
vrp_backward_gk_positive
rv_gk_22d_forward_ann_label
vrp_forward_expost_gk_label
feature_allowed
```

## Robustness Columns

These may appear when the corresponding Phase 2 RV columns are available:

```text
rv_cc_22d_ann_lag1
vrp_backward_cc
vrp_backward_cc_positive
rv_parkinson_22d_ann_lag1
vrp_backward_parkinson
vrp_backward_parkinson_positive
rv_rs_22d_ann_lag1
vrp_backward_rs
vrp_backward_rs_positive
rv_yz_22d_ann_lag1
vrp_backward_yz
vrp_backward_yz_positive
```

Robustness columns are diagnostic-only in Phase 3. They are not primary live features.

## Feature / Label Boundary

Primary feature-like columns:

```text
iv_ann
rv_gk_22d_ann_lag1
vrp_backward_gk
vrp_backward_gk_positive
```

Label-only columns:

```text
rv_gk_22d_forward_ann_label
vrp_forward_expost_gk_label
```

Forbidden live-feature substrings:

```text
future
forward
expost
label
```

Any column containing those substrings must not be used as a tradable feature, regime input, or strategy signal.

## Forward Ex-Post Label Convention

```text
rv_gk_22d_forward_ann_label_t =
    252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})
```

```text
vrp_forward_expost_gk_label_t =
    iv_ann_t - rv_gk_22d_forward_ann_label_t
```

These columns use future data and are outcome/evaluation labels only.

## Calendar Mismatch Report

Local path:

```text
reports/tables/calendar_mismatches.csv
```

Expected columns:

```text
market
iv_start
iv_end
rv_start
rv_end
iv_rows
rv_rows
common_dates
iv_only_dates
rv_only_dates
first_iv_only_date
first_rv_only_date
```

Policy:

```text
IV and RV are aligned by exact-date inner join only.
No forward-fill or backward-fill is allowed.
```

## No-Lookahead Artifact Rules

- `vrp_backward_gk` is point-in-time safe.
- `rv_gk_22d_forward_ann_label` is an outcome label only.
- `vrp_forward_expost_gk_label` is an outcome label only.
- Robustness VRP columns are diagnostic-only.
- `feature_allowed` is based on registered primary feature columns only.
- No generated Phase 3 artifact should be used to bypass the feature registry.

## Sensitivity and Reproducibility Notes

- IV panels depend on Phase 1 cleaned VIX / India VIX files.
- VRP panels depend on Phase 2 RV files.
- `VIX^2` and `India VIX^2` are implied variance proxies, not direct variance swap quotes.
- 22 trading days approximates a 30-calendar-day VIX-style horizon.
- Generated artifacts should be regenerated locally rather than committed.

## Lightweight Review Packet

Use these instead of sending large generated files:

```text
terminal output:
python scripts/build_features.py --market ALL --feature vrp

terminal output:
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py

preview:
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/calendar_mismatches.csv
```

Do not send full parquet files unless explicitly requested.
