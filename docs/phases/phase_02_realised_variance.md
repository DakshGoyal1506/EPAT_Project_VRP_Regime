# Phase 2 — Realised Variance Construction

## Status

`complete / frozen`

Phase 2 is implemented, tested, and frozen. Do not rewrite working logic unless there is a correctness, no-lookahead, reproducibility, or documentation bug.

## Objective

Construct realised variance panels for the US and India from clean daily OHLC data produced by Phase 1.

This phase creates the realised-variance side of the VRP pipeline. Phase 3 consumes these RV panels when constructing implied variance and variance risk premium.

## Phase Boundary

Phase 2 includes:

- daily return construction
- daily realised variance estimators
- rolling 22-trading-day annualised realised variance
- realised variance diagnostics
- estimator comparison plots
- tests for estimator formulas and no-lookahead rolling behavior

Phase 2 does not include:

- implied variance
- VIX / India VIX conversion
- variance risk premium
- HAR-RV forecasting
- regimes
- strategies
- backtests
- broker logic

## Inputs from Phase 1

```text
data/processed/us_underlying.parquet
data/processed/india_underlying.parquet
```

Expected input columns:

```text
date
open
high
low
close
```

Optional columns such as `volume`, `adj_close`, `source`, and `symbol` may be present but are not required for the realised-variance formulas.

## Files Owned by Phase 2

```text
src/vrp/features/returns.py
src/vrp/features/realized_variance.py
src/vrp/features/feature_io.py
src/vrp/reports/rv_diagnostics.py
scripts/build_features.py
tests/test_rv_estimators.py
```

`src/vrp/features/README.md` documents the broader feature package shared by Phase 2 and Phase 3.

## Main Functions

### `src/vrp/features/returns.py`

```text
compute_log_returns(df, price_col="close")
compute_simple_returns(df, price_col="close")
add_gap_return(df)
add_intraday_return(df)
add_all_returns(df)
```

Return conventions:

```text
log_return_t = log(close_t / close_{t-1})
simple_return_t = close_t / close_{t-1} - 1
gap_return_t = log(open_t / close_{t-1})
intraday_return_t = log(close_t / open_t)
```

### `src/vrp/features/realized_variance.py`

```text
validate_ohlc(df)
close_to_close_daily_var(df)
parkinson_daily_var(df)
garman_klass_daily_var(df)
rogers_satchell_daily_var(df)
yang_zhang_rolling_var(df, window=22)
rolling_realized_variance(df, var_col, window=22)
annualize_variance(var, periods=252)
annualize_vol(var, periods=252)
build_rv_panel(df, market, symbol, window=22, annualization_periods=252)
```

### `src/vrp/reports/rv_diagnostics.py`

```text
make_rv_summary(panels, window=22)
make_estimator_correlations(panels, window=22)
write_rv_metadata(...)
plot_rv_estimators(...)
write_rv_diagnostics(...)
```

## Realised Variance Estimators

All estimator outputs are variance, not volatility. All ratios use natural logarithms.

### Close-to-close

```text
rv_cc_daily_t = [log(close_t / close_{t-1})]^2
```

Purpose:

* baseline estimator
* uses only close-to-close returns
* easy to audit and explain

Limitation:

* ignores intraday high-low movement
* can understate volatility when intraday range is large but close-to-close movement is small

### Parkinson

```text
rv_parkinson_daily_t = [log(high_t / low_t)]^2 / [4 * log(2)]
```

Purpose:

* range-based estimator
* uses daily high and low
* captures intraday movement ignored by close-to-close variance

Limitation:

* ignores open and close
* does not directly handle overnight jumps

### Garman-Klass

```text
rv_gk_daily_t =
    0.5 * [log(high_t / low_t)]^2
    - (2 * log(2) - 1) * [log(close_t / open_t)]^2
```

Purpose:

* primary realised-variance estimator for the project
* uses open, high, low, and close
* more informative than close-to-close when only daily OHLC data is available

Limitation:

* still an estimator, not directly observed realised variance
* sensitive to bad OHLC values
* not a substitute for intraday realised variance

### Rogers-Satchell

```text
rv_rs_daily_t =
    log(high_t / open_t) * log(high_t / close_t)
    + log(low_t / open_t) * log(low_t / close_t)
```

Purpose:

* OHLC estimator
* useful as a robustness check against Garman-Klass
* handles drift differently from Parkinson and Garman-Klass

Limitation:

* still depends on daily OHLC assumptions
* can differ materially during trending markets

### Yang-Zhang

Yang-Zhang is implemented as a rolling estimator only. No `rv_yz_daily` column is created.

Conceptual components:

```text
open_return_t = log(open_t / close_{t-1})
close_return_t = log(close_t / open_t)

yz_rolling_var_t =
    rolling_var(open_return)
    + k * rolling_var(close_return)
    + (1 - k) * rolling_mean(Rogers-Satchell component)
```

Purpose:

* accounts for overnight gap and intraday movement
* used as a robustness estimator

Important project convention:

```text
rv_yz_daily is forbidden by design
```

Yang-Zhang is treated as a rolling-window estimator, not a one-row daily estimator.

## Primary Estimator

The primary realised variance estimator for the project is:

```text
rv_gk_22d_ann
```

Reason:

* Garman-Klass uses the full daily OHLC bar.
* It is richer than close-to-close.
* It remains transparent and implementable from public daily data.
* Other estimators are retained for robustness and diagnostic comparison.

## Rolling and Annualisation Convention

Rolling realised variance uses a trailing rolling mean of daily variance:

```text
rolling_rv_t = mean(daily_var_{t-window+1}, ..., daily_var_t)
```

Default window:

```text
window = 22 trading days
```

Annualised variance:

```text
rv_*_22d_ann = 252 * rolling_mean(daily_variance, 22)
```

The project uses annualised variance, not annualised volatility, for VRP comparison.

Reason:

```text
VIX² and India VIX² are annualised implied variance proxies.
The realised variance side must also be annualised.
```

## Output Column Contract

Phase 2 RV panels contain:

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

There is intentionally no:

```text
rv_yz_daily
```

## Point-in-Time Feature Safety

The rolling columns are trailing and use only observations available up to the current row.

At date `t`:

```text
rv_*_22d_ann_t uses daily variance from t-21 through t
```

It does not use:

```text
t+1
t+2
...
future observations
```

Phase 2 itself does not create trading features or future labels. Phase 3 handles explicit lagging before VRP feature use.

## Outcome / Label Columns

Phase 2 does not create outcome labels.

No Phase 2 column should contain:

```text
future
forward
expost
label
```

Forward realised variance labels are Phase 3-owned.

## Generated Local Outputs

```text
data/processed/us_rv.parquet
data/processed/india_rv.parquet

reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json

reports/figures/rv_estimators_us.png
reports/figures/rv_estimators_india.png
```

These are generated artifacts and are local-only by default.

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
data/processed/us_rv.parquet
data/processed/india_rv.parquet
reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json
reports/figures/rv_estimators_us.png
reports/figures/rv_estimators_india.png
```

Reason:

* generated from deterministic commands
* can be regenerated locally
* may be large or environment-dependent
* not needed in Git history

## Commands

Build Phase 2 outputs:

```bash
python scripts/build_features.py --market ALL --feature rv --window 22
```

Run Phase 2 tests:

```bash
pytest tests/test_rv_estimators.py
```

Optional full validation:

```bash
pytest
```

## Validation Checklist

* [ ] `data/processed/us_rv.parquet` is generated locally.
* [ ] `data/processed/india_rv.parquet` is generated locally.
* [ ] `rv_gk_22d_ann` exists in both panels.
* [ ] `rv_yz_22d_ann` exists in both panels.
* [ ] `rv_yz_daily` does not exist.
* [ ] Daily estimator outputs are variance, not volatility.
* [ ] Rolling RV uses trailing windows only.
* [ ] Annualised RV uses 252 periods.
* [ ] The default rolling window is 22 trading days.
* [ ] RV summary diagnostics are generated locally.
* [ ] RV estimator correlation diagnostics are generated locally.
* [ ] RV estimator figures are generated locally.
* [ ] `pytest tests/test_rv_estimators.py` passes.

## No-Lookahead Checklist

* [ ] No centered rolling windows.
* [ ] No future daily variance used in rolling RV.
* [ ] No future/outcome/label columns created in Phase 2.
* [ ] No trading signal created in Phase 2.
* [ ] Phase 3 is responsible for explicit lagging before VRP feature use.

## Known Limitations

* Daily OHLC estimators are proxies for true realised variance.
* Intraday realised variance is not computed in this phase.
* Range-based estimators are sensitive to bad high/low data.
* Close-to-close variance may miss intraday volatility.
* Parkinson may miss overnight jumps.
* Yang-Zhang is rolling-only; no daily Yang-Zhang estimator is emitted.
* The 22-trading-day window approximates a one-month horizon but is not identical to 30 calendar days.
* Phase 2 does not compare RV to VIX or India VIX. That belongs to Phase 3.

## Review Checklist

For code review, inspect:

```text
src/vrp/features/returns.py
src/vrp/features/realized_variance.py
src/vrp/features/feature_io.py
src/vrp/reports/rv_diagnostics.py
scripts/build_features.py
tests/test_rv_estimators.py
```

For generated-output review, send lightweight substitutes:

```text
terminal output from:
python scripts/build_features.py --market ALL --feature rv --window 22

terminal output from:
pytest tests/test_rv_estimators.py

preview of:
reports/tables/rv_summary.csv
reports/tables/rv_estimator_correlations.csv
reports/tables/rv_metadata.json
```

Do not send full parquet panels unless explicitly requested.
