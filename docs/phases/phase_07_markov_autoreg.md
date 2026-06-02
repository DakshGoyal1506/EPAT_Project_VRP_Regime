# Phase 7 - Markov Autoregression / AR-Aware Regime Model

## 1. Status

Status: complete / frozen after documentation cleanup.

Phase 7 implements a point-in-time-safe Markov autoregression regime layer for the EPAT VRP regime project.

Primary implementation:

```text
statsmodels MarkovAutoregression
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

K=2 is the primary model. K=3 is robustness only unless explicitly promoted in a later documented review.

## 2. Objective

Phase 7 upgrades the Phase 6 Gaussian HMM regime layer by adding a model that directly accounts for observed-series autocorrelation.

The Gaussian HMM models regime-dependent emissions but does not directly model autoregression in the observed VRP/RV series. Markov autoregression adds AR dynamics to the regime process, allowing the state model to capture persistence and mean-reverting behavior in the observed target itself.

Phase 7 produces filtered regime probabilities and next-session economic regime labels for later strategy and backtest phases.

## 3. Phase Boundary

Phase 7 owns:

```text
Markov autoregression regime fitting
train-only target transformation
fit-status firewall
full-series filtering using train-fitted parameters
diagnostic smoothing
economic state mapping
MAR diagnostics and no-lookahead audits
```

Phase 7 does not own:

```text
strategy exposure rules
backtest accounting
transaction costs
robustness portfolio analysis
cross-market lead-lag analysis
broker paper signals
paper execution
live trading
```

HMM and threshold states are allowed only for diagnostic comparison after MAR states are assigned. They must not be used as MAR model inputs.

## 4. Files Owned by Phase 7

```text
configs/model_markov_autoreg.yaml
scripts/train_markov_autoreg.py
src/vrp/regimes/markov_autoreg.py
src/vrp/regimes/markov_autoreg_registry.py
src/vrp/reports/markov_autoreg_diagnostics.py
tests/test_markov_autoreg.py
tests/test_markov_autoreg_no_lookahead.py
```

Optional future/stub file if explicitly added later:

```text
src/vrp/regimes/dynamax_arhmm.py
```

Dynamax/JAX is not required for Phase 7. Any Dynamax AR-HMM route is optional and must remain stub-only unless explicitly enabled in a later scoped task.

## 5. Main Functions, Classes, and Scripts

### Script

```text
scripts/train_markov_autoreg.py
```

Main responsibilities:

```text
parse Phase 7 CLI arguments
load config
run one market or all markets
run primary model or candidate grid
write model-specific outputs
write primary alias outputs
write model payloads
write metadata, audits, and diagnostics
```

### Registry

```text
src/vrp/regimes/markov_autoreg_registry.py
```

Important objects and utilities:

```text
MARModelSpec
MARConfig
load_markov_autoreg_config
expand_candidate_specs
resolve_target_column
processed_input_path
model_specific_output_path
primary_alias_output_path
model_specific_model_path
primary_alias_model_path
forbidden input-column helpers
```

### Core Model Module

```text
src/vrp/regimes/markov_autoreg.py
```

Important objects:

```text
MARPreparedData
MARCandidateFit
MARFullFilterResult
MARSignalOutput
MARFitFirewallSummary
MARProbabilityAudit
MARParameterLookaheadAudit
MAREconomicStateMapping
```

Important functions:

```text
prepare_mar_data_from_config
prepare_mar_model_data
fit_apply_target_transform_train_only
fit_markov_autoreg_candidate
filter_full_series_with_train_params
align_mar_probabilities_to_eligible_frame
audit_aligned_probabilities
build_parameter_lookahead_audit
build_mar_signal_output
build_train_state_economic_summary
label_mar_states_economically
add_next_session_signal_columns
validate_mar_signal_output
```

### Diagnostics Module

```text
src/vrp/reports/markov_autoreg_diagnostics.py
```

Important functions:

```text
write_phase7_diagnostics
build_mar_duration_summary
build_mar_state_by_year
build_hmm_agreement_table
build_threshold_agreement_table
build_regime_model_comparison_row
```

## 6. Config Files Used

Primary config:

```text
configs/model_markov_autoreg.yaml
```

The config defines:

```text
primary K=2 MAR model
candidate grid
target-column mapping
target availability rules
train/test split
train-only target transform
fit firewall
validation thresholds
forbidden input policy
probability columns
audit fields
output paths
diagnostic paths
```

Primary target mapping:

```text
vrp_har -> vrp_har_gk
```

Primary target availability rule:

```text
har_forecast_available == true
```

Primary target transform:

```text
winsorize_train_quantiles_then_standardize
lower_quantile = 0.005
upper_quantile = 0.995
```

The winsorization caps, mean, and standard deviation are estimated on the train window only.

## 7. Input Files

Required local inputs:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Optional diagnostic comparison inputs:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
```

The HMM and threshold files are diagnostic-only inputs and must not affect MAR fitting.

## 8. Generated Output Files

Primary alias outputs:

```text
data/processed/us_markov_autoreg_regimes.parquet
data/processed/india_markov_autoreg_regimes.parquet
```

Model-specific outputs:

```text
data/processed/markov_autoreg/us_markov_autoreg_vrp_har_order1_k2_sv.parquet
data/processed/markov_autoreg/india_markov_autoreg_vrp_har_order1_k2_sv.parquet
data/processed/markov_autoreg/*.parquet
```

Model payloads:

```text
models/us_markov_autoreg.pkl
models/india_markov_autoreg.pkl
models/markov_autoreg/*.pkl
```

Diagnostics:

```text
reports/tables/phase_7/us/mar_metadata.json
reports/tables/phase_7/india/mar_metadata.json
reports/tables/phase_7/us/mar_candidate_model_ranking.csv
reports/tables/phase_7/india/mar_candidate_model_ranking.csv
reports/tables/phase_7/us/mar_state_summary.csv
reports/tables/phase_7/india/mar_state_summary.csv
reports/tables/phase_7/us/mar_transition_matrix.csv
reports/tables/phase_7/india/mar_transition_matrix.csv
reports/tables/phase_7/us/mar_ar_stability.csv
reports/tables/phase_7/india/mar_ar_stability.csv
reports/tables/phase_7/us/mar_probability_audit.csv
reports/tables/phase_7/india/mar_probability_audit.csv
reports/tables/phase_7/us/mar_no_lookahead_audit.csv
reports/tables/phase_7/india/mar_no_lookahead_audit.csv
reports/tables/phase_7/us/mar_duration_summary.csv
reports/tables/phase_7/india/mar_duration_summary.csv
reports/tables/phase_7/us/mar_state_by_year.csv
reports/tables/phase_7/india/mar_state_by_year.csv
reports/tables/phase_7/us/mar_hmm_agreement.csv
reports/tables/phase_7/india/mar_hmm_agreement.csv
reports/tables/phase_7/us/mar_threshold_agreement.csv
reports/tables/phase_7/india/mar_threshold_agreement.csv
reports/tables/regime_model_comparison.csv
```

Figures, if generated later:

```text
reports/figures/phase_7/*
```

## 9. Committed vs Local-Only Outputs

Commit:

```text
source code
configuration
scripts
tests
documentation
README files
.gitkeep placeholders
```

Keep local-only:

```text
data/processed/*.parquet
data/processed/markov_autoreg/*.parquet
models/*.pkl
models/markov_autoreg/*.pkl
reports/tables/phase_7/*
reports/figures/phase_7/*
logs
broker cache
```

Generated MAR panels and model binaries should not be committed.

## 10. Commands to Regenerate Outputs

Primary Phase 7 run:

```bash
python scripts/train_markov_autoreg.py --market US --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market INDIA --target vrp_har --order 1 --states 2 --primary --force
python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force
```

Approved candidate grid:

```bash
python scripts/train_markov_autoreg.py --market ALL --run-grid --force
```

CLI help:

```bash
python scripts/train_markov_autoreg.py --help
```

## 11. Tests to Run

Phase-specific tests:

```bash
pytest tests/test_markov_autoreg.py tests/test_markov_autoreg_no_lookahead.py
```

Global smoke test:

```bash
pytest
```

## 12. Validation Checklist

```text
[ ] Primary K=2 MAR model fits for US.
[ ] Primary K=2 MAR model fits for India.
[ ] Target transform parameters are estimated on train only.
[ ] Full-series filtering uses train-fitted parameters only.
[ ] First AR warmup row is unavailable.
[ ] Filtered probabilities sum to 1 after warmup.
[ ] K=2 transition probability is 0.0 only after valid model rows.
[ ] Smoothed probabilities are diagnostic-only.
[ ] HMM states are not used as MAR inputs.
[ ] Threshold states are not used as MAR inputs.
[ ] Forward/ex-post/future/label columns are not used as MAR inputs.
[ ] Primary alias outputs are written locally.
[ ] Model-specific outputs are written locally.
[ ] No-lookahead audit passes.
[ ] Probability audit passes.
[ ] AR-stability table is written.
[ ] State summary is written.
[ ] Duration summary is written.
[ ] State-by-year summary is written.
[ ] HMM agreement table is written if Phase 6 outputs exist.
[ ] Threshold agreement table is written if Phase 5 outputs exist.
[ ] tests pass.
```

## 13. No-Lookahead and Safety Rules

1. Fit MAR parameters on the chronological train window only.
2. Estimate winsorization caps, target mean, and target standard deviation on the train window only.
3. Apply train-fitted parameters to the full eligible series using `filter()`.
4. Use filtered probabilities for backtest-facing regime decisions.
5. Keep smoothed probabilities diagnostic-only.
6. Do not fill AR warmup rows with calm, transition, stress, or zero probabilities.
7. Do not use HMM states, HMM probabilities, threshold states, crisis labels, forward labels, ex-post labels, or future columns as MAR inputs.
8. HMM/threshold agreement is diagnostic only and must not select the MAR model.
9. No strategy, backtest, or broker action belongs in Phase 7.

## 14. Known Limitations

1. Markov autoregression fitting can be numerically fragile.
2. The target transform uses train-only winsorization and standardization to stabilize statsmodels fitting.
3. K=3 is not primary and can invent weak transition states.
4. State labels are economic interpretations, not observed truth.
5. The MAR stress state may have elevated `vrp_har_gk` mean when implied variance rises sharply during volatility stress.
6. HMM/threshold agreement can be low without invalidating MAR.
7. Diagnostic smoothed probabilities are not valid backtest signals.
8. Generated MAR outputs are local-only and must be regenerated by reviewers if needed.

## 15. Review Checklist

```text
[ ] docs/phases/phase_07_markov_autoreg.md exists.
[ ] docs/artifacts/phase_07_artifacts.md exists.
[ ] src/vrp/regimes/README.md documents Phase 7.
[ ] docs/commands.md contains Phase 7 commands.
[ ] docs/artifact_inventory.md has complete Phase 7 artifact rows.
[ ] reports README files mention Phase 7 local-only outputs.
[ ] .gitignore blocks MAR panels, model binaries, logs, and generated reports.
[ ] git ls-files check shows no generated MAR parquet/model/log/env files.
[ ] git diff --check passes.
[ ] git status --short shows only intended docs/config cleanup files.
```
