# `vrp.regimes`

## Purpose

This package contains regime-detection logic for the EPAT VRP regime project.

It covers:

```text
Phase 5  - deterministic threshold baseline regimes
Phase 6  - Gaussian HMM regimes
Phase 7  - Markov autoregression / AR-HMM-style regime upgrade
Phase 8+ - robustness or diagnostic regime integrations where applicable
```

## Phase Ownership

### Phase 5 Ownership

Phase 5 owns:

```text
threshold.py
regime_registry.py
canonical economic state IDs in state_labeling.py
```

Phase 5 state mapping:

```text
0 = calm
1 = transition
2 = stress
```

### Phase 6 Ownership - Gaussian HMM

Phase 6 owns:

```text
hmm_registry.py
hmm_validation.py
online_filter.py
hmm_features.py
hmm_scaling.py
gaussian_hmm.py
state_labeling.py HMM extension section
```

Phase 6 responsibilities:

```text
define approved HMM feature sets
block forward/ex-post/threshold/crisis/HMM-derived leakage features
construct eligible HMM feature panels
fit scalers on the chronological train window only
fit Gaussian HMM candidates on the train window only
compute custom point-in-time filtered probabilities
keep smoothed probabilities diagnostic-only
map raw HMM states to calm/transition/stress using train-period economic properties
validate candidates for occupancy, transition, covariance, probability, and economic interpretability
write HMM diagnostics and no-lookahead audits
```

Phase 6 must not:

```text
construct strategy exposure
run PnL backtests
use threshold_state as an HMM feature or target
use crisis windows as state labels
use forward/ex-post labels as HMM features
use full-sample smoothed probabilities as tradable signals
place or preview broker orders
```

### Shared/Later-Phase Ownership

Later phases may extend regime modules with HMM or Markov-specific functions. Those later extensions must not change the canonical economic state IDs.

`state_labeling.py` is intentionally shared: Phase 5 owns the canonical constants and threshold labels, while Phase 6 extends the file with HMM economic-state mapping helpers.

## Responsibilities

This package is responsible for:

1. Building timestamp-safe regime features.
2. Rejecting forbidden forward/ex-post/label construction columns.
3. Building deterministic threshold regimes.
4. Building probabilistic regime models in later phases.
5. Mapping model states into economic states.
6. Preserving no-lookahead boundaries.

## Main Modules

### `state_labeling.py`

Defines canonical economic state IDs and mappings:

```text
CALM = 0
TRANSITION = 1
STRESS = 2
```

Also contains shared state-label validation and mapping helpers.

### `regime_registry.py`

Defines approved regime construction features and forbidden feature substrings.

Forbidden in construction:

```text
future
forward
expost
label
```

HAR-derived features are valid only when `har_forecast_available == True`.

### `threshold.py`

Builds Phase 5 threshold regimes.

Main steps:

1. Load and validate threshold config.
2. Validate input panel.
3. Compute strict-prior rolling thresholds.
4. Build five component states:
   - IV percentile state
   - RV percentile state
   - drawdown state
   - IV slope state
   - HAR-VRP z-score state
5. Combine components into final threshold state.
6. Write blocked reasons and trigger reasons.

## Expected Inputs

Phase 5 expects Phase 4 HAR-VRP panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Core construction columns:

```text
iv_ann
iv_close
rv_gk_22d_ann_lag1
vrp_backward_gk
har_rv_gk_22d_forecast_ann
vrp_har_gk
har_forecast_available
```

Optional drawdown columns:

```text
close
adj_close
underlying_close
log_return
simple_return
```

Phase 6 expects Phase 4 HAR-VRP panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Phase 6 may also read Phase 5 threshold panels for diagnostic comparison only:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
```

## Expected Outputs

Phase 5 generated outputs are local-only by default:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
reports/tables/threshold_*.csv
reports/tables/threshold_*.json
reports/figures/threshold_*.png
```

Phase 6 generated outputs are local-only by default:

```text
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
data/processed/*_hmm_*.parquet
models/us_gaussian_hmm.pkl
models/india_gaussian_hmm.pkl
models/*_hmm_*.pkl
reports/tables/phase_6/us/*
reports/tables/phase_6/india/*
reports/figures/phase_6/*
```

## Commands

Threshold regime CLI:

```bash
python scripts/train_regimes.py --model threshold --market US --force
python scripts/train_regimes.py --model threshold --market INDIA --force
python scripts/train_regimes.py --model threshold --market ALL --force
```

Gaussian HMM CLI:

```bash
python scripts/train_regimes.py --market US --model gaussian_hmm --primary --force
python scripts/train_regimes.py --market INDIA --model gaussian_hmm --primary --force
python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force
```

Help:

```bash
python scripts/train_regimes.py --help
```

## Tests

Phase 5 tests:

```bash
pytest tests/test_threshold_regimes.py
pytest tests/test_regime_no_lookahead.py
pytest tests/test_no_lookahead.py
```

Phase 6 tests:

```bash
pytest tests/test_hmm_filtering.py
pytest tests/test_hmm_scaling.py
pytest tests/test_hmm_model.py
pytest tests/test_hmm_no_lookahead.py
pytest tests/test_no_lookahead.py
```

## Safety and No-Lookahead Boundaries

1. Do not use future realised variance as a construction feature.
2. Do not use forward/ex-post labels as regime construction features.
3. Do not use full-sample percentiles for backtest-facing labels.
4. Thresholds at date `t` must use only prior history.
5. HAR-derived regime features require `har_forecast_available == True`.
6. Do not fill/backfill unavailable component states.
7. Crisis windows are diagnostics-only.
8. Full-sample smoothed HMM probabilities are diagnostic-only and must not become tradable signals.
9. Tradable HMM/Markov decisions must use filtered probabilities available at time `t`.

## What This Module Must Not Do

This module must not:

1. Download raw market data.
2. Manually repair production data.
3. Create strategy exposure rules in Phase 5.
4. Run backtests.
5. Place broker orders.
6. Use crisis windows as training labels.
7. Use forward/ex-post labels as regime features.
8. Pool US and India thresholds unless an explicit later phase defines such analysis.
