# Strategy Signal Module

## Purpose

`src/vrp/strategies/` owns Phase 9 strategy signal construction.

It converts already-produced feature, forecast, and regime panels into next-session short-vol exposure intentions.

It does not run backtests, compute returns, estimate transaction costs, place broker orders, or perform cross-market strategy construction.

## Phase Ownership

Primary phase:

```text
Phase 9 - strategy signal construction
```

Downstream consumers:

```text
Phase 10 - vectorised research backtest and robustness
Phase 11 - IBKR paper-signal readiness layer
Phase 13 - cross-market US-India analysis
```

## Responsibilities

This module is responsible for:

```text
loading and validating Phase 9 strategy config
locking the approved strategy universe
pure exposure rule functions
probability validation
carry-aware signal gating
no-lookahead column sanitation
long-format signal panel schema validation
next-session timing rules
```

This module is not responsible for:

```text
PnL
returns
Sharpe ratio
drawdown
transaction costs
performance ranking
option-chain execution
live trading
IBKR order placement
MSVOL strategy construction
cross-market strategy construction
```

## Approved Strategy Universe

Exactly seven active strategies are allowed:

```text
unconditional_full
threshold_hard_filter
threshold_defensive
hmm_prob_linear
hmm_prob_linear_carry
mar_prob_linear
mar_prob_linear_carry
```

Rejected/deferred strategies:

```text
threshold_carry_aware
hmm_hard
hmm_defensive
mar_hard
mar_defensive
probability_product
probability_cutoff_filter
msvol
msgarch
```

## Main Modules

| Module | Purpose |
|---|---|
| `strategy_config.py` | Load and validate `configs/strategies.yaml`. |
| `strategy_registry.py` | Lock approved strategies, model families, and forbidden inputs. |
| `exposure_rules.py` | Pure exposure-decision functions. |
| `signal_schema.py` | Canonical output schema and no-lookahead sanitation. |
| `signal_builder.py` | Build long-format Phase 9 signal panels. |
| `cross_market_overlay.py` | Phase 13 analysis-only overlay diagnostic; not part of Phase 9 strategy construction. |

## Expected Inputs

Phase 9 expects these local generated panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
```

MSVOL / MSGARCH files are not valid Phase 9 strategy inputs.

## Expected Outputs

```text
data/processed/us_strategy_signals.parquet
data/processed/india_strategy_signals.parquet

reports/tables/phase_9/strategy_signal_summary.csv
reports/tables/phase_9/strategy_exposure_by_year.csv
reports/tables/phase_9/strategy_exposure_change_summary.csv
reports/tables/phase_9/strategy_blocked_reason_summary.csv
reports/tables/phase_9/strategy_no_lookahead_audit.csv
reports/tables/phase_9/strategy_metadata.json
```

All generated outputs are local-only unless explicitly selected later as final-report artifacts.

## Output Schema

The canonical Phase 9 output is long format.

One row per:

```text
market
signal_observation_date
target_trade_date
strategy_name
```

Canonical exposure column:

```text
target_exposure
```

Exposure convention:

```text
-1.0 = full short-vol exposure
 0.0 = flat / no exposure
```

No long-vol exposure is allowed.

## Missing Versus Valid Flat Exposure

Unavailable input:

```text
strategy_available = False
target_exposure = NaN
decision_reason = unavailable
```

Valid flat decision:

```text
strategy_available = True
target_exposure = 0.0
blocked_reason = none
```

Examples of valid flat reasons:

```text
stress_veto
stress_defensive_flat
negative_or_zero_vrp_har
stress_probability_veto
probability_linear
probability_linear_carry
```

## Carry Gate

Carry-aware strategies use:

```text
har_forecast_available is True
vrp_har_gk > 0
p_stress <= 0.40
```

The numeric `vrp_har_gk > 0` test is the source of truth.

`vrp_har_gk_positive` is audit-only if present.

## Timing Rules

Threshold:

```text
state observed at date t
signal available after date t close
target_trade_date = next available trading date
```

HMM and MAR:

```text
use *_for_next_session columns
use *_signal_trade_date
do not double-shift
```

## No-lookahead Boundaries

Forbidden strategy inputs include:

```text
forward realised variance labels
ex-post VRP labels
future/outcome/label columns
smoothed HMM/MAR probabilities
crisis-window labels
MSVOL/MSGARCH fields
```

Allowed:

```text
HMM/MAR filtered next-session-safe signal columns
```

## Commands

CLI help:

```bash
python scripts/build_signals.py --help
```

Build all Phase 9 signals:

```bash
python scripts/build_signals.py --market ALL --strategy all --force
```

Build one market:

```bash
python scripts/build_signals.py --market US --strategy all --force
python scripts/build_signals.py --market INDIA --strategy all --force
```

Single-strategy smoke checks:

```bash
python scripts/build_signals.py --market ALL --strategy hmm_prob_linear --force
python scripts/build_signals.py --market ALL --strategy mar_prob_linear --force
python scripts/build_signals.py --market ALL --strategy threshold_hard_filter --force
```

## Tests

```bash
pytest tests/test_exposure_rules.py
pytest tests/test_signal_builder.py
pytest tests/test_strategy_no_lookahead.py
pytest tests/test_phase9_diagnostics.py
```

Combined:

```bash
pytest tests/test_exposure_rules.py tests/test_signal_builder.py tests/test_strategy_no_lookahead.py tests/test_phase9_diagnostics.py
```

## What This Module Must Not Do

```text
compute realised performance
rank strategies by performance
consume future or ex-post labels
consume smoothed probabilities as strategy signals
read MSVOL files as strategy inputs
create hidden strategy variants
place or preview broker orders
write large artifacts to Git
```

## Phase 13 Analysis-Only Overlay Exception

`cross_market_overlay.py` belongs to Phase 13, not Phase 9.

It may:

```text
read locked Phase 9 India strategy signals
read locked Phase 10 India backtest panels
apply a lagged-US-stress exposure block for analysis only
write Phase 13 overlay diagnostics
```

It must not:

```text
modify Phase 9 signal files
modify Phase 10 backtest files
create a new approved strategy
change the seven-strategy Phase 9 universe
read Phase 11 broker or paper-signal artifacts
place or preview broker orders
```

All Phase 13 overlay outputs are marked `analysis_only = true`.
