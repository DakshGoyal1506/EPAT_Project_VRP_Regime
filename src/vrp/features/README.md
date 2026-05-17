# VRP Features

This package contains the Phase 2 and Phase 3 feature-construction logic for the EPAT VRP project.

It is the canonical home for reusable feature engineering code. The modules here are designed to be imported by scripts, tests, notebooks, and later model or strategy code. They do not contain research notebooks, ad hoc analysis, or backtest logic.

## Purpose

The features package builds the time series that feed the rest of the project:

- realised variance estimators from OHLC data
- implied variance from VIX / India VIX levels
- calendar alignment between IV and RV panels
- VRP construction with strict no-lookahead rules
- registry metadata that separates live features, labels, and robustness diagnostics

## Modules

### `calendars.py`

Calendar alignment and mismatch reporting utilities.

Responsibilities:

- validate that panels contain a `date` column
- reject invalid or duplicate dates
- align IV and RV series using an inner join on exact dates
- report date coverage mismatches between IV and RV panels
- write calendar mismatch diagnostics to CSV

Use this module when you need to understand whether two panels overlap cleanly before constructing VRP.

### `feature_io.py`

Shared load/save helpers for feature panels.

Responsibilities:

- read and write canonical processed feature files
- ensure parent directories exist before writing
- validate required columns before downstream processing

### `feature_registry.py`

Defines the project feature registry.

This module acts as a guardrail against lookahead leakage. It distinguishes:

- primary live feature columns
- label-only columns
- robustness diagnostic columns
- forbidden substrings that should never appear in live features

Current registry policy:

- primary live features remain GK-based
- forward or ex-post label columns are not live features
- robustness columns are diagnostic only and do not enter `VRP_FEATURE_COLUMNS`

### `implied_variance.py`

Builds annualised implied variance from VIX-style index levels.

Main steps:

- infer the close column from common aliases
- validate the index level values
- enforce positive, finite, non-decimal-scaled VIX readings
- compute `iv_ann = (iv_close / 100)^2`

### `realized_variance.py`

Builds realised variance estimators from OHLC data.

Supported estimators:

- close-to-close
- Parkinson
- Garman-Klass
- Rogers-Satchell
- Yang-Zhang rolling variance

The Garman-Klass estimate is the primary Phase 3 live RV feature used in the VRP pipeline. The other estimators remain available as robustness diagnostics.

### `returns.py`

Return helpers used by realised variance construction.

Responsibilities:

- compute log returns
- compute simple returns
- add gap and intraday return series
- keep return logic centralized for Phase 2 feature generation

### `vrp.py`

Constructs the Phase 3 VRP panel.

Responsibilities:

- merge IV and RV on common dates
- compute backward GK VRP
- compute backward VRP robustness columns for other RV estimators
- compute GK-only forward ex-post labels
- flag rows that are eligible for live feature use
- return a stable ordered output panel

Important rules:

- GK remains the primary live feature path
- robustness columns are diagnostic only
- forward/ex-post labels are GK-only
- no forward-filling is performed

## Typical Workflow

The usual feature flow is:

1. build RV panels from OHLC data
2. build IV panels from VIX / India VIX data
3. align IV and RV by date
4. build VRP panels with live and diagnostic columns
5. write diagnostics and reports

## Common Imports

Typical entry points used elsewhere in the repository:

- `vrp.features.realized_variance.build_rv_panel`
- `vrp.features.implied_variance.build_implied_variance`
- `vrp.features.calendars.align_market_dates`
- `vrp.features.vrp.build_vrp_panel`
- `vrp.features.feature_registry.make_vrp_feature_metadata`

## Validation

Run the relevant tests from the repository root:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py
```

## Output Contract

The main Phase 3 VRP panel is expected to retain the following live columns:

- `iv_ann`
- `rv_gk_22d_ann_lag1`
- `vrp_backward_gk`
- `vrp_backward_gk_positive`
- `rv_gk_22d_forward_ann_label`
- `vrp_forward_expost_gk_label`

Robustness columns may also be present when the relevant RV estimators exist, but they remain outside the primary live feature registry.
