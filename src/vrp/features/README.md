# VRP Features

This package contains the reusable Phase 2 and Phase 3 feature-construction logic. It is the canonical home for feature engineering code used by scripts, tests, notebooks, and later research layers.

Phase definitions and generated artifact rules live in `docs/`.

## Responsibilities

- build realised variance estimators from OHLC data
- build implied variance from VIX-style levels
- align IV and RV panels on exact market dates
- construct VRP with strict no-lookahead rules
- maintain the feature registry that separates live features, labels, and diagnostics

## Modules

### `calendars.py`

Validates and aligns IV / RV panels using exact-date inner joins. It also reports calendar mismatches and writes mismatch diagnostics.

### `feature_io.py`

Shared load/save helpers for canonical processed feature files.

### `feature_registry.py`

Defines the anti-lookahead firewall for Phase 3.

- `VRP_FEATURE_COLUMNS` are the primary live features.
- `VRP_LABEL_COLUMNS` are labels only.
- `VRP_ROBUSTNESS_COLUMNS` are diagnostic only.
- forbidden substrings such as `future`, `forward`, `expost`, and `label` are blocked from live features.

### `implied_variance.py`

Builds annualised implied variance.

Main contract:

- infer the close column from common aliases
- validate VIX-style levels for numeric, finite, positive, non-decimal-scaled values
- compute `iv_ann = (iv_close / 100)^2`

### `realized_variance.py`

Builds realised variance estimators from OHLC data.

Supported estimators include close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, and Yang-Zhang. Garman-Klass is the primary Phase 3 live RV path.

### `returns.py`

Shared return helpers for realised variance construction.

### `vrp.py`

Constructs the Phase 3 VRP panel.

Key behavior:

- merge IV and RV on common dates only
- compute GK backward VRP as the primary live spread
- compute backward robustness VRP columns for non-primary estimators
- compute GK-only forward ex-post labels
- flag rows that are eligible for live feature use

## Output Contract

The primary live Phase 3 VRP panel keeps these columns stable:

- `iv_ann`
- `rv_gk_22d_ann_lag1`
- `vrp_backward_gk`
- `vrp_backward_gk_positive`
- `rv_gk_22d_forward_ann_label`
- `vrp_forward_expost_gk_label`

Robustness columns may appear when the corresponding RV estimators are available, but they stay out of `VRP_FEATURE_COLUMNS`.

## Validation

Run the feature-focused test slice from the repository root:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_vrp_alignment.py tests/test_no_lookahead.py
```

## HAR / Phase 4 note

The feature layer feeds the Phase 4 HAR engine. After building and validating VRP panels, the HAR forecasting module in `src/vrp/forecasting/` is used to construct prospective HAR forecasts and HAR-based VRP outputs that are consumed by reporting and downstream regime/backtest components.
