# `vrp.regimes`

## Purpose

This package contains regime-detection logic for the EPAT VRP regime project.

It covers:

```text
Phase 5  - deterministic threshold baseline regimes
Phase 6  - Gaussian HMM regimes
Phase 7  - Markov autoregression / AR-HMM-style regime upgrade
Phase 8  - Python-only MSVOL robustness appendix
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

### Phase 7 Ownership - Markov Autoregression

Phase 7 owns:

```text
markov_autoreg_registry.py
markov_autoreg.py
```

Phase 7 report diagnostics live in:

```text
src/vrp/reports/markov_autoreg_diagnostics.py
```

Phase 7 responsibilities:

```text
define Markov autoregression model specs
resolve MAR target columns
block forbidden HMM/threshold/crisis/future/forward/ex-post/label inputs
prepare the primary vrp_har_gk target
estimate target transform parameters on train only
fit statsmodels MarkovAutoregression on train only
apply full-series Hamilton filtering using train-fitted parameters only
keep smoothed probabilities diagnostic-only
align AR(1) probabilities back to dates
blank AR warmup rows
map raw MAR states into calm/transition/stress using train-period economic properties
write MAR diagnostics and no-lookahead audits
```

Primary model:

```text
target = vrp_har
target_col = vrp_har_gk
order = 1
n_states = 2
switching_ar = true
switching_trend = true
switching_variance = true
```

K=2 is primary. K=3 is robustness only unless explicitly promoted in a later documented review.

Phase 7 must not:

```text
construct strategy exposure
run backtests
use HMM states as model inputs
use threshold states as model inputs
use crisis windows as labels
use forward/ex-post/future/label columns as model inputs
use full-sample smoothed probabilities as tradable signals
place or preview broker orders
```

Dynamax/JAX AR-HMM support is optional and stub-only unless explicitly enabled later.

### Phase 8 Ownership - Python-only MSVOL Robustness Appendix

Phase 8 owns:

```text
msvol_model.py
msvol_adapter.py
```

Related Phase 8 report modules live in:

```text
src/vrp/reports/msvol_diagnostics.py
src/vrp/reports/msvol_no_lookahead.py
```

Phase 8 responsibilities:

```text
export return-only inputs through the legacy msgarch-named export script
fit a Python-only Markov-switching volatility model
use AR(1)-prefiltered index returns
extract filtered probabilities
keep smoothed probabilities diagnostic-only
map lower-variance state to calm
map higher-variance state to stress
write standardized processed MSVOL regime panels
write probability audits and metadata
write diagnostics comparing MSVOL stress regimes with threshold/HMM/MAR regimes
write no-lookahead audits
```

Active model:

```text
implementation = statsmodels MarkovRegression
k_regimes = 2
switching_variance = true
true_msgarch = false
```

Phase 8 must not:

```text
claim the Python model is true MSGARCH
construct strategy exposure
run backtests
produce VaR / ES
perform cross-market lead-lag analysis
use HAR residuals as the active model input
use VRP_HAR as the active model target
use future/outcome/label columns as inputs
use smoothed probabilities as tradable signals
place or preview broker orders
```

True R MSGARCH remains optional/future only.

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

Phase 7 expects Phase 4 HAR-VRP panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Phase 7 may read Phase 5 and Phase 6 outputs for diagnostic comparison only:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
```

Phase 8 expects Phase 4 HAR-VRP panels for return export:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

It writes legacy-named input CSVs:

```text
data/interim/msgarch/us_msgarch_input.csv
data/interim/msgarch/india_msgarch_input.csv
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

Phase 7 generated outputs are local-only by default:

```text
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
data/processed/markov_autoreg/*.parquet
models/us_markov_autoreg.pkl
models/india_markov_autoreg.pkl
models/markov_autoreg/*.pkl
reports/tables/phase_7/us/*
reports/tables/phase_7/india/*
reports/figures/phase_7/*
```

Phase 8 generated outputs are local-only by default:

```text
data/interim/msvol/*
data/processed/us_msvol_regimes.parquet
data/processed/india_msvol_regimes.parquet
reports/tables/phase_8/*
reports/figures/phase_8/*
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

Markov autoregression CLI:

```bash
python scripts/train_markov_autoreg.py --market US --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market INDIA --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market ALL --run-grid --force
python scripts/train_markov_autoreg.py --help
```

MSVOL CLI:

```bash
python scripts/export_msgarch_inputs.py --market ALL
python scripts/run_msvol_regimes.py --market ALL
python scripts/import_msvol_outputs.py --market ALL
python scripts/run_msvol_diagnostics.py --market ALL
python scripts/run_msvol_no_lookahead_audit.py --market ALL
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

Phase 7 tests:

```bash
pytest tests/test_markov_autoreg.py
pytest tests/test_markov_autoreg_no_lookahead.py
```

Phase 8 tests:

```bash
pytest tests/test_msgarch_export.py
pytest tests/test_msvol_model.py
pytest tests/test_msvol_adapter.py
pytest tests/test_msvol_diagnostics.py
pytest tests/test_msvol_no_lookahead.py
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
10. MSVOL is diagnostic-only and not true MSGARCH.
11. MSVOL smoothed probabilities are diagnostic-only.
12. MSVOL outputs must not feed Phase 9 strategy construction or backtests.

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
