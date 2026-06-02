# `vrp.features`

## Purpose

This package contains the reusable Phase 2 and Phase 3 feature-construction logic for the EPAT VRP regime project.

It is the canonical home for:

- return construction
- realised variance construction
- implied variance construction
- IV/RV calendar alignment
- variance risk premium construction
- feature/label/robustness registry
- no-lookahead feature safeguards

Production feature logic must live here or in adjacent `src/vrp/` modules, not inside notebooks.

## Phase Ownership

| Phase | Scope |
|---:|---|
| 2 | Realised variance construction from OHLC data |
| 3 | Implied variance and VRP construction |

Phase 4+ may consume outputs from this package, but Phase 4+ model logic should not be documented as Phase 2/3 ownership.

## Responsibilities

This package must:

- validate feature inputs
- compute returns and realised variance consistently
- compute implied variance from VIX-style index levels
- align IV and RV panels with exact-date inner joins
- compute backward point-in-time VRP
- compute explicitly labelled forward ex-post evaluation labels
- separate live features from labels and robustness diagnostics
- avoid silent forward-fill or backward-fill
- expose stable columns for downstream forecasting, regimes, and backtests

This package must not:

- download raw data
- train HAR models
- train HMM or AR-HMM regimes
- generate strategy signals
- run backtests
- place or preview broker orders
- let future/outcome/label columns enter live feature lists

## Main Modules

| Module | Phase | Purpose |
|---|---:|---|
| `returns.py` | 2 | Log returns, simple returns, gap returns, intraday returns |
| `realized_variance.py` | 2 | Close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, rolling annualised RV |
| `feature_io.py` | 2/3 | Shared feature-panel save/load helpers |
| `implied_variance.py` | 3 | VIX / India VIX close to annualised implied variance |
| `calendars.py` | 3 | IV/RV exact-date alignment and calendar mismatch reporting |
| `feature_registry.py` | 3 | Anti-lookahead registry for features, labels, robustness columns |
| `vrp.py` | 3 | Backward VRP, forward ex-post labels, robustness VRP diagnostics |
| `src/vrp/reports/rv_diagnostics.py` | 2 | RV summary, correlations, metadata, figures |
| `src/vrp/reports/vrp_diagnostics.py` | 3 | VRP summary, metadata, IV/RV/VRP figures |

## Expected Phase 1 Inputs

Phase 2 consumes:

```text
data/processed/us_underlying.parquet
data/processed/india_underlying.parquet
```

Phase 3 consumes:

```text
data/processed/us_vix.parquet
data/processed/india_vix.parquet
data/processed/us_rv.parquet
data/processed/india_rv.parquet
```

## Expected Phase 2 Outputs

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet

reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json

reports/figures/rv_estimators_us.png
reports/figures/rv_estimators_india.png
```

These are generated local artifacts and are not committed by default.

## Expected Phase 3 Outputs

```text
data/processed/us_iv.parquet
data/processed/india_iv.parquet
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet

reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/calendar_mismatches.csv

reports/figures/us_iv_rv_vrp.png
reports/figures/india_iv_rv_vrp.png
```

These are generated local artifacts and are not committed by default.

## Phase 2 Column Contract

RV panels contain:

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

Primary realised variance column:

```text
rv_gk_22d_ann
```

Forbidden by design:

```text
rv_yz_daily
```

Yang-Zhang is rolling-only in this project.

## Phase 3 Column Contract

IV panels contain:

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

VRP panels contain primary columns:

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

Robustness columns may include:

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

## Commands

Build Phase 2 RV:

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

Build Phase 3 IV:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build Phase 3 VRP:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

Phase 3 VRP currently supports only:

```text
window = 22
horizon = 22
```

Reason:

```text
The Phase 3 feature registry and label names are fixed to 22-day columns.
```

## Tests

Phase 2:

```bash
pytest tests/test_rv_estimators.py
```

Phase 3:

```bash
pytest tests/test_implied_variance.py
pytest tests/test_calendar_alignment.py
pytest tests/test_vrp_alignment.py
pytest tests/test_no_lookahead.py
pytest tests/test_build_features_cli.py
```

Combined Phase 2/3 slice:

```bash
pytest tests/test_rv_estimators.py tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Full suite:

```bash
pytest
```

## No-Lookahead Rules

1. Rolling RV must be trailing only.
2. Backward VRP must use lagged RV.
3. Same-day RV must not be used as the primary tradable VRP input.
4. Forward ex-post VRP labels are outcomes only.
5. Columns containing `future`, `forward`, `expost`, or `label` must not enter live feature sets.
6. `VRP_FEATURE_COLUMNS` is the approved primary live feature list.
7. `VRP_LABEL_COLUMNS` is evaluation-only.
8. `VRP_ROBUSTNESS_COLUMNS` is diagnostic-only for Phase 3.
9. IV and RV panels must be aligned by exact-date inner join only.
10. No IV/RV forward-fill or backward-fill is allowed.

## Feature Registry Contract

Primary live features:

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

Robustness-only diagnostics:

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

## Notebook Boundary

Notebooks may read and inspect generated Phase 2/3 artifacts.

Notebooks must not:

- define production formulas
- overwrite canonical outputs as part of normal review
- replace tested modules
- introduce untracked feature logic

## Generated Artifact Boundary

Generated parquet panels, CSV/JSON diagnostics, and figures stay local unless explicitly approved as final-report artifacts.

Do not commit by default:

```text
data/processed/*.parquet
reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
reports/figures/*.svg
```

## Safety Boundary

This package does not contain live-trading logic.

Broker execution, order placement, paper-trading adapters, regime signals, and strategy backtests are outside Phase 2/3 feature construction.

iv_ann = (iv_close / 100) ** 2
```

VRP panels contain primary columns:

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

Robustness columns may include:

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

## Commands

Build Phase 2 RV:

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

Build Phase 3 IV:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build Phase 3 VRP:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

Phase 3 VRP currently supports only:

```text
window = 22
horizon = 22
```

Reason:

```text
The Phase 3 feature registry and label names are fixed to 22-day columns.
```

## Tests

Phase 2:

```bash
pytest tests/test_rv_estimators.py
```

Phase 3:

```bash
pytest tests/test_implied_variance.py
pytest tests/test_calendar_alignment.py
pytest tests/test_vrp_alignment.py
pytest tests/test_no_lookahead.py
pytest tests/test_build_features_cli.py
```

Combined Phase 2/3 slice:

```bash
pytest tests/test_rv_estimators.py tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Full suite:

```bash
pytest
```

## No-Lookahead Rules

1. Rolling RV must be trailing only.
2. Backward VRP must use lagged RV.
3. Same-day RV must not be used as the primary tradable VRP input.
4. Forward ex-post VRP labels are outcomes only.
5. Columns containing `future`, `forward`, `expost`, or `label` must not enter live feature sets.
6. `VRP_FEATURE_COLUMNS` is the approved primary live feature list.
7. `VRP_LABEL_COLUMNS` is evaluation-only.
8. `VRP_ROBUSTNESS_COLUMNS` is diagnostic-only for Phase 3.
9. IV and RV panels must be aligned by exact-date inner join only.
10. No IV/RV forward-fill or backward-fill is allowed.

## Feature Registry Contract

Primary live features:

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

Robustness-only diagnostics:

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

## Notebook Boundary

Notebooks may read and inspect generated Phase 2/3 artifacts.

Notebooks must not:

- define production formulas
- overwrite canonical outputs as part of normal review
- replace tested modules
- introduce untracked feature logic

## Generated Artifact Boundary

Generated parquet panels, CSV/JSON diagnostics, and figures stay local unless explicitly approved as final-report artifacts.

Do not commit by default:

```text

reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
reports/figures/*.svg
```

## Safety Boundary

This package does not contain live-trading logic.

Broker execution, order placement, paper-trading adapters, regime signals, and strategy backtests are outside Phase 2/3 feature construction.

## Build Commands

Run these commands from the repository root to rebuild Phase 2/3 feature artifacts (RV, IV, VRP):

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
python scripts/build_features.py --market ALL --feature iv
python scripts/build_features.py --market ALL --feature vrp
```
