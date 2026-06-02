# Phase 09 - Strategy Signal Construction

## Status

Complete / frozen.

Phase 9 converts Phase 4/5/6/7 outputs into next-session short-vol exposure intentions. It does not run a backtest and does not evaluate strategy performance.

## Objective

Construct a fixed, interpretable family of short-vol strategy signal rules that map regime outputs and prospective HAR-based VRP information into next-session target exposure intentions.

Exposure convention:

```text
target_exposure = -1.0  -> full short-vol exposure
target_exposure =  0.0  -> flat / no short-vol exposure
```

No long-vol exposure is allowed in Phase 9.

## Phase Boundary

Phase 9 owns:

```text
strategy rule definitions
exposure-intention construction
long-format signal panel construction
signal availability diagnostics
no-lookahead column sanitation for strategy inputs
Phase 9 metadata
```

Phase 9 does not own:

```text
realised returns
PnL
Sharpe ratio
drawdown
transaction costs
backtest ranking
strategy performance interpretation
broker execution
IBKR paper-signal readiness
cross-market US-India analysis
MSVOL strategy usage
```

Phase 10 owns backtesting. Phase 11 owns broker paper-signal readiness. Phase 13 owns cross-market analysis.

## Files Owned by This Phase

```text
configs/strategies.yaml
scripts/build_signals.py
src/vrp/strategies/__init__.py
src/vrp/strategies/strategy_config.py
src/vrp/strategies/strategy_registry.py
src/vrp/strategies/exposure_rules.py
src/vrp/strategies/signal_schema.py
src/vrp/strategies/signal_builder.py
src/vrp/reports/strategy_diagnostics.py
tests/test_exposure_rules.py
tests/test_signal_builder.py
tests/test_strategy_no_lookahead.py
tests/test_phase9_diagnostics.py
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

Rejected/deferred variants:

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

## Strategy Rules

### unconditional_full

```text
target_exposure = -1.0
```

### threshold_hard_filter

```text
if threshold_state_name == "stress":
    target_exposure = 0.0
else:
    target_exposure = -1.0
```

### threshold_defensive

```text
calm       -> -1.0
transition -> -0.25
stress     -> 0.0
```

### hmm_prob_linear

```text
target_exposure = -clip(
    hmm_filtered_prob_calm_for_next_session
    - hmm_filtered_prob_stress_for_next_session,
    0.0,
    1.0,
)
```

### hmm_prob_linear_carry

Base rule:

```text
target_exposure = -clip(p_calm - p_stress, 0.0, 1.0)
```

Carry gate:

```text
if har_forecast_available is not True:
    strategy_available = False

elif vrp_har_gk <= 0:
    strategy_available = True
    target_exposure = 0.0

elif p_stress > 0.40:
    strategy_available = True
    target_exposure = 0.0

else:
    use probability-linear exposure
```

The carry gate uses numeric `vrp_har_gk > 0` as the source of truth. The boolean `vrp_har_gk_positive` is audit-only if present.

### mar_prob_linear

```text
target_exposure = -clip(
    mar_filtered_prob_calm_for_next_session
    - mar_filtered_prob_stress_for_next_session,
    0.0,
    1.0,
)
```

### mar_prob_linear_carry

Same rule as `hmm_prob_linear_carry`, using MAR filtered probabilities.

## Main Functions, Classes, and Scripts

### `src/vrp/strategies/strategy_config.py`

Loads and validates `configs/strategies.yaml`.

Key objects:

```text
ExposureBounds
TimingPolicy
FrozenConstants
StrategyDefinition
StrategyConfig
load_strategy_config()
validate_strategy_config()
get_strategy_definitions()
get_market_input_paths()
get_market_output_path()
strategy_config_hash()
```

### `src/vrp/strategies/strategy_registry.py`

Locks the strategy universe and no-lookahead strategy-input policy.

Key objects:

```text
APPROVED_STRATEGY_NAMES
REJECTED_STRATEGY_NAMES
ALLOWED_STRATEGY_MODELS
FORBIDDEN_STRATEGY_MODELS
assert_no_strategy_forbidden_columns()
assert_no_msvol_strategy_use()
validate_strategy_names()
validate_strategy_model_map()
```

### `src/vrp/strategies/exposure_rules.py`

Pure exposure decision rules.

Key objects:

```text
ExposureDecision
ProbabilityValidationError
clip_short_vol_exposure()
unconditional_full_exposure()
threshold_hard_filter_exposure()
threshold_defensive_exposure()
validate_probability_triplet()
probability_linear_exposure()
probability_linear_decision()
apply_probability_carry_gate()
```

### `src/vrp/strategies/signal_schema.py`

Canonical Phase 9 schema and no-lookahead input sanitation.

Key objects:

```text
PHASE9_OUTPUT_COLUMNS
sanitize_strategy_input_frame()
sanitize_input_frames()
validate_phase9_signal_panel()
build_no_lookahead_audit_records()
```

### `src/vrp/strategies/signal_builder.py`

Builds the long-format Phase 9 signal panel.

Key objects:

```text
SignalBuildResult
build_phase9_signal_panel()
default_strategy_definitions()
```

### `src/vrp/reports/strategy_diagnostics.py`

Creates signal-only diagnostics and metadata.

Key functions:

```text
create_strategy_signal_summary()
create_strategy_exposure_by_year()
create_strategy_exposure_change_summary()
create_strategy_blocked_reason_summary()
create_no_lookahead_audit_table()
build_phase9_metadata()
create_all_phase9_diagnostics()
write_diagnostic_tables()
write_metadata_json()
```

### `scripts/build_signals.py`

CLI entry point:

```bash
python scripts/build_signals.py --market ALL --strategy all --force
```

## Config Files Used

```text
configs/strategies.yaml
```

Important frozen constants:

```text
min_exposure = -1.0
max_exposure = 0.0
transition_exposure = -0.25
stress_probability_cutoff = 0.40
probability_sum_tolerance = 0.001
primary_probability_rule = calm_minus_stress
carry_gate_vrp_column = vrp_har_gk
carry_gate_threshold = 0.0
```

## Input Files

Phase 9 reads:

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

Phase 9 must not read:

```text
data/processed/*msvol*
data/processed/*msgarch*
```

## Generated Output Files

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

Optional future local figures:

```text
reports/figures/phase_9/**/*
```

## Commit Policy

Commit:

```text
code
config
scripts
tests
docs
README files
.gitkeep placeholders
```

Local-only:

```text
data/processed/*strategy_signals*.parquet
reports/tables/phase_9/*.csv
reports/tables/phase_9/*.json
reports/figures/phase_9/*
```

## Commands to Regenerate Outputs

```bash
python scripts/build_signals.py --market US --strategy all --force
python scripts/build_signals.py --market INDIA --strategy all --force
python scripts/build_signals.py --market ALL --strategy all --force
```

Single-strategy smoke examples:

```bash
python scripts/build_signals.py --market ALL --strategy hmm_prob_linear --force
python scripts/build_signals.py --market ALL --strategy mar_prob_linear --force
python scripts/build_signals.py --market ALL --strategy threshold_hard_filter --force
```

## Tests to Run

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

Full suite:

```bash
pytest
```

## Validation Checklist

Phase 9 is valid only if:

```text
pytest passes
python scripts/build_signals.py --market ALL --strategy all --force succeeds
exactly seven strategies appear
no rejected strategies appear
target_exposure is always between -1.0 and 0.0 for available rows
unavailable rows have NaN target_exposure
available rows have blocked_reason = none
threshold signals are shifted to next available trading date
HMM signals are not double-shifted
MAR signals are not double-shifted
forbidden_columns_used = []
MSVOL policy is excluded_diagnostic_only
no PnL/returns/cost/Sharpe/drawdown fields are created
```

## No-lookahead and Safety Rules

1. Do not use forward realised variance labels as strategy inputs.
2. Do not use ex-post VRP labels as strategy inputs.
3. Do not use future/outcome/label columns as tradable features.
4. Do not use smoothed HMM/MAR probabilities as backtest-facing signals.
5. Use only filtered probabilities available at time `t`.
6. Do not use MSVOL or MSGARCH as Phase 9 strategy models.
7. Do not use crisis-window labels as strategy inputs.
8. Do not apply same-day signals to same-day returns.
9. Do not create backtest metrics in Phase 9.
10. Do not add hidden strategy variants before Phase 10 evaluation.

## Missing Versus Valid Flat Decision

Unavailable means required inputs are missing or invalid:

```text
strategy_available = False
target_exposure = NaN
decision_reason = unavailable
```

Valid flat means the rule was evaluated and intentionally chose zero exposure:

```text
strategy_available = True
target_exposure = 0.0
blocked_reason = none
```

Valid flat reasons include:

```text
stress_veto
stress_defensive_flat
negative_or_zero_vrp_har
stress_probability_veto
probability_linear
probability_linear_carry
```

The `probability_linear` and `probability_linear_carry` flat cases occur when `p_calm <= p_stress`, so `clip(p_calm - p_stress, 0, 1) = 0`.

## Known Limitations

1. Phase 9 creates proxy exposure intentions, not executable option trades.
2. Exposure units are research-layer short-vol proxies.
3. Strategy rules are intentionally simple and fixed before performance evaluation.
4. Carry-aware rules use `vrp_har_gk > 0`; no z-score, rolling median, or optimized buffer is added in this phase.
5. Probability sizing depends on calibrated filtered probabilities from upstream models.
6. Missing upstream regimes or probabilities create unavailable signals.
7. Phase 9 diagnostics describe signal behaviour only and are not evidence of profitability.

## Review Checklist

Before freezing Phase 9, confirm:

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
pytest tests/test_exposure_rules.py tests/test_signal_builder.py tests/test_strategy_no_lookahead.py tests/test_phase9_diagnostics.py
```

Expected `git ls-files` match:

```text
.env.example
```

No generated parquet/model/log/env files should be tracked.
