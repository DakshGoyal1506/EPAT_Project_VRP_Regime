# Phase 3 — Implied Variance and VRP Construction

## Status

`complete / frozen`

Phase 3 is implemented, tested, and frozen. Do not rewrite working logic unless there is a correctness, leakage, safety, reproducibility, or documentation bug.

## Objective

Construct implied variance and variance risk premium panels for the US and India in a point-in-time safe way.

This phase converts VIX / India VIX close levels into annualised implied variance, aligns implied variance with realised variance from Phase 2, computes backward VRP features, creates forward ex-post VRP labels, and enforces feature/label separation.

## Phase Boundary

Phase 3 includes:

- VIX / India VIX to implied variance conversion
- IV/RV exact-date alignment
- calendar mismatch reporting
- backward Garman-Klass VRP
- backward robustness VRPs for non-primary realised-variance estimators
- Garman-Klass-only forward ex-post VRP labels
- feature/label/robustness registry
- VRP diagnostics and metadata
- no-lookahead tests

Phase 3 does not include:

- HAR-RV forecasting
- forecast-based VRP
- threshold regimes
- HMM regimes
- AR-HMM / Markov autoregression
- strategies
- backtests
- broker logic

## Inputs

From Phase 1:

```text
data/processed/us_vix.parquet
data/processed/india_vix.parquet
```

From Phase 2:

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet
```

## Files Owned by Phase 3

```text
src/vrp/features/implied_variance.py
src/vrp/features/calendars.py
src/vrp/features/feature_registry.py
src/vrp/features/vrp.py
src/vrp/reports/vrp_diagnostics.py
scripts/build_features.py
scripts/build_features_cli.py

tests/test_implied_variance.py
tests/test_calendar_alignment.py
tests/test_vrp_alignment.py
tests/test_no_lookahead.py
tests/test_build_features_cli.py
```

Note: `src/vrp/features/vrp.py` may also contain later HAR-related helper logic. That helper is Phase 4-owned and is not part of Phase 3 scope.

## Main Functions

### `src/vrp/features/implied_variance.py`

```text
infer_iv_close_column(vix_df)
validate_vix_values(df, iv_col="iv_close")
build_implied_variance(vix_df, market, ...)
```

### `src/vrp/features/calendars.py`

```text
align_market_dates(iv_df, rv_df)
report_calendar_mismatches(iv_df, rv_df, market)
build_calendar_mismatch_table(rows)
write_calendar_mismatch_report(rows, output_path)
```

### `src/vrp/features/feature_registry.py`

```text
VRP_FEATURE_COLUMNS
VRP_LABEL_COLUMNS
VRP_ROBUSTNESS_COLUMNS
FORBIDDEN_FEATURE_SUBSTRINGS
assert_registry_is_valid()
assert_no_lookahead_feature_columns(feature_columns)
make_vrp_feature_metadata()
```

### `src/vrp/features/vrp.py`

```text
merge_iv_rv(iv_df, rv_df, market)
compute_backward_vrp(panel, rv_col="rv_gk_22d_ann")
compute_backward_vrp_robustness(panel)
compute_forward_expost_vrp(panel, rv_daily_col="rv_gk_daily", horizon=22)
flag_feature_columns_vs_label_columns(panel)
build_vrp_panel(iv_df, rv_df, market, horizon=22)
```

### `src/vrp/reports/vrp_diagnostics.py`

```text
make_vrp_summary(panels)
write_vrp_metadata(...)
plot_iv_rv_vrp(...)
write_vrp_diagnostics(...)
```

## VIX to Implied Variance Convention

Input close column:

```text
iv_close
```

For US:

```text
iv_close = VIX close
```

For India:

```text
iv_close = India VIX close
```

Formula:

```text
iv_ann = (iv_close / 100) ** 2
```

Meaning:

```text
iv_ann = annualised implied variance proxy
```

VIX-style levels are treated as annualised volatility percentages. The implied-variance module rejects non-numeric, missing, non-finite, non-positive, extreme, and decimal-scaled VIX values.

## IV Panel Output Contract

```text
date
market
iv_symbol
iv_close
iv_ann
```

Generated local outputs:

```text
data/processed/us_iv.parquet
data/processed/india_iv.parquet
```

## IV/RV Alignment Rule

IV and RV panels are aligned by exact common dates only:

```text
inner join on date
```

Forbidden:

```text
forward-fill IV
forward-fill RV
backward-fill IV
backward-fill RV
outer join for VRP construction
cross-market calendar merge
```

Calendar mismatches are reported separately in:

```text
reports/tables/calendar_mismatches.csv
```

Calendar mismatch report columns:

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

## Backward VRP Construction

Primary realised variance column:

```text
rv_gk_22d_ann
```

Point-in-time lagged RV:

```text
rv_gk_22d_ann_lag1_t = rv_gk_22d_ann_{t-1}
```

Primary backward VRP:

```text
vrp_backward_gk_t = iv_ann_t - rv_gk_22d_ann_lag1_t
```

Positivity flag:

```text
vrp_backward_gk_positive = vrp_backward_gk > 0
```

These columns are feature-like and point-in-time safe.

## Forward Ex-Post Label Construction

Future realised variance label:

```text
rv_gk_22d_forward_ann_label_t =
    252 * mean(rv_gk_daily_{t+1}, ..., rv_gk_daily_{t+22})
```

Forward ex-post VRP label:

```text
vrp_forward_expost_gk_label_t =
    iv_ann_t - rv_gk_22d_forward_ann_label_t
```

These columns use future data and are outcome/evaluation labels only.

They must never enter strategies, regimes, or live feature sets.

## Feature Registry

Primary live feature columns:

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

Robustness-only diagnostic columns:

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

Forbidden live-feature substrings:

```text
future
forward
expost
label
```

Any column containing those substrings is forbidden from live feature lists.

## Robustness Extension

Garman-Klass remains the primary VRP estimator.

Backward robustness VRPs are computed for:

```text
close-to-close
Parkinson
Rogers-Satchell
Yang-Zhang
```

These robustness columns are diagnostic-only and excluded from `VRP_FEATURE_COLUMNS`.

No forward ex-post robustness labels are created in Phase 3.

## VRP Panel Output Contract

Primary columns:

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

Optional robustness columns appear when available:

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

Generated local outputs:

```text
data/processed/us_vrp.parquet
data/processed/india_vrp.parquet
```

## Generated Diagnostics

```text
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/calendar_mismatches.csv

reports/figures/us_iv_rv_vrp.png
reports/figures/india_iv_rv_vrp.png
```

## Commands

Build IV panels:

```bash
python scripts/build_features.py --market ALL --feature iv
```

Build VRP panels and diagnostics:

```bash
python scripts/build_features.py --market ALL --feature vrp
```

Run Phase 3 tests:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Optional full validation:

```bash
pytest
```

## CLI Boundary

Phase 3 VRP currently supports only:

```text
window = 22
horizon = 22
```

Reason:

```text
The Phase 3 feature registry and label names are fixed to 22-day columns.
```

Non-22 VRP windows/horizons should be implemented only as a later explicit extension with dynamic naming and metadata.

## Commit vs Local-Only Policy

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

Reason:

- generated from deterministic commands
- can be regenerated locally
- may be large or environment-dependent
- not needed in Git history

## No-Lookahead Checklist

- [ ] Backward VRP uses `rv_gk_22d_ann_lag1`, not same-day RV.
- [ ] Forward ex-post labels use future RV and are clearly labelled.
- [ ] `VRP_FEATURE_COLUMNS` contains no `future`, `forward`, `expost`, or `label` columns.
- [ ] `VRP_LABEL_COLUMNS` are excluded from features.
- [ ] `VRP_ROBUSTNESS_COLUMNS` are excluded from primary live features.
- [ ] `feature_allowed` is based only on primary live features.
- [ ] IV/RV alignment uses exact-date inner join only.
- [ ] No IV or RV forward-fill is used.
- [ ] No cross-market calendar merge is used.
- [ ] VRP CLI rejects non-22 window/horizon for Phase 3 fixed-column registry.
- [ ] Forward ex-post labels are not used in strategies, regimes, or backtests.

## Known Limitations

- VIX and India VIX are proxies for implied variance, not direct variance swap quotes.
- `VIX^2` and `India VIX^2` are approximations.
- 22 trading days approximates the 30-calendar-day VIX-style horizon.
- Forward ex-post labels are not tradable.
- Robustness VRP columns are descriptive/diagnostic only in Phase 3.
- Phase 3 does not forecast future realised variance; HAR-RV belongs to Phase 4.
- Phase 3 does not assign regimes or produce strategy signals.
- Calendar mismatches are reported, not repaired by fill logic.

## Review Checklist

For code review, inspect:

```text
src/vrp/features/implied_variance.py
src/vrp/features/calendars.py
src/vrp/features/feature_registry.py
src/vrp/features/vrp.py
src/vrp/reports/vrp_diagnostics.py
scripts/build_features.py
scripts/build_features_cli.py

tests/test_implied_variance.py
tests/test_calendar_alignment.py
tests/test_vrp_alignment.py
tests/test_no_lookahead.py
tests/test_build_features_cli.py
```

For generated-output review, send lightweight substitutes:

```text
terminal output from:
python scripts/build_features.py --market ALL --feature vrp

terminal output from:
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py

preview of:
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/tables/calendar_mismatches.csv
```

Do not send full parquet panels unless explicitly requested.
