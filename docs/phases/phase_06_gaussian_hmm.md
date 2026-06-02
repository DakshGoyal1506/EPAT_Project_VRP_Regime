# Phase 6 - Gaussian HMM Regime Model

## 1. Status

Complete / frozen.

Phase 6 implements the Gaussian Hidden Markov Model latent-regime baseline for the EPAT VRP regime project.

Selected primary model after full candidate-grid runs:

```text
US:    F3, K=3, covariance=diag
India: F3, K=3, covariance=diag
```

## 2. Objective

Estimate latent VRP/volatility regimes for US and Indian markets using a Gaussian HMM while preserving strict no-lookahead boundaries.

The HMM is used as a probabilistic regime model. It produces filtered state probabilities and economic state labels that later phases may consume as regime inputs.

## 3. Phase Boundary

Phase 6 includes:

```text
Gaussian HMM feature construction
train-only scaling
train-only model fitting
candidate grid fitting
custom point-in-time forward filtering
diagnostic-only smoothed probabilities
raw-state to economic-state mapping
candidate validation and rejection
HMM diagnostic tables
HMM no-lookahead audit
```

Phase 6 excludes:

```text
strategy construction
position sizing
PnL backtesting
robustness backtests
Markov autoregression
MSVOL / MSGARCH
broker or IBKR logic
cross-market analysis
```

## 4. Files Owned by This Phase

Config:

```text
configs/model_hmm.yaml
```

Script branch:

```text
scripts/train_regimes.py --model gaussian_hmm
```

Source modules:

```text
src/vrp/regimes/hmm_registry.py
src/vrp/regimes/hmm_validation.py
src/vrp/regimes/online_filter.py
src/vrp/regimes/hmm_features.py
src/vrp/regimes/hmm_scaling.py
src/vrp/regimes/gaussian_hmm.py
src/vrp/regimes/state_labeling.py
src/vrp/reports/hmm_diagnostics.py
```

Tests:

```text
tests/test_hmm_filtering.py
tests/test_hmm_scaling.py
tests/test_hmm_model.py
tests/test_hmm_no_lookahead.py
tests/test_no_lookahead.py
```

## 5. Main Functions / Classes / Scripts

Key classes:

```text
HMMCandidateSpec
HMMFitConfig
HMMCandidateFitResult
HMMCandidateOutput
HMMFilterResult
HMMFeaturePanel
HMMScaledFeaturePanel
HMMStateLabelingResult
HMMValidationRules
```

Key functions:

```text
build_hmm_feature_panel
scale_hmm_feature_panel
fit_hmm_candidate
build_hmm_candidate_output
fit_and_build_hmm_candidate_output
forward_filter_gaussian
label_hmm_states_from_train_properties
build_hmm_probability_audit_table
build_hmm_no_lookahead_audit_table
```

CLI entry point:

```bash
python scripts/train_regimes.py --model gaussian_hmm
```

## 6. Config Files Used

```text
configs/model_hmm.yaml
```

Important config sections:

```text
primary_model
fallback_model
candidate_models
train_test_split
hmm_fit
feature_sets
feature_construction
conditional_features
forbidden_regime_construction_features
scaling
model_validation
state_labeling
probability_outputs
signal_availability
paths
```

Primary model:

```text
feature_set: F3
n_states: 3
covariance_type: diag
```

Fallback model:

```text
feature_set: F3
n_states: 2
covariance_type: diag
```

Candidate grid:

```text
F1-F4 x K={2,3} x covariance={diag,full}
```

## 7. Input Files

Phase 6 consumes Phase 4 HAR-VRP panels:

```text
data/processed/us_vrp_har.parquet
data/processed/india_vrp_har.parquet
```

Optional diagnostic comparison inputs from Phase 5:

```text
data/processed/us_threshold_regimes.parquet
data/processed/india_threshold_regimes.parquet
```

The threshold-regime inputs are diagnostic-only and are not used as HMM features, labels, model-selection targets, or state-mapping inputs.

## 8. Generated Output Files

Primary alias outputs:

```text
data/processed/us_hmm_regimes.parquet
data/processed/india_hmm_regimes.parquet
models/us_gaussian_hmm.pkl
models/india_gaussian_hmm.pkl
```

Model-specific outputs:

```text
data/processed/*_hmm_*.parquet
models/*_hmm_*.pkl
```

Diagnostic outputs:

```text
reports/tables/phase_6/us/hmm_candidate_model_ranking.csv
reports/tables/phase_6/us/hmm_feature_availability.csv
reports/tables/phase_6/us/hmm_state_summary.csv
reports/tables/phase_6/us/hmm_transition_matrix.csv
reports/tables/phase_6/us/hmm_state_duration_summary.csv
reports/tables/phase_6/us/hmm_state_by_year.csv
reports/tables/phase_6/us/hmm_threshold_agreement.csv
reports/tables/phase_6/us/hmm_crisis_hit_table.csv
reports/tables/phase_6/us/hmm_crisis_lead_lag_table.csv
reports/tables/phase_6/us/hmm_forward_label_by_state.csv
reports/tables/phase_6/us/hmm_probability_audit.csv
reports/tables/phase_6/us/hmm_no_lookahead_audit.csv
reports/tables/phase_6/us/hmm_metadata.json

reports/tables/phase_6/india/hmm_candidate_model_ranking.csv
reports/tables/phase_6/india/hmm_feature_availability.csv
reports/tables/phase_6/india/hmm_state_summary.csv
reports/tables/phase_6/india/hmm_transition_matrix.csv
reports/tables/phase_6/india/hmm_state_duration_summary.csv
reports/tables/phase_6/india/hmm_state_by_year.csv
reports/tables/phase_6/india/hmm_threshold_agreement.csv
reports/tables/phase_6/india/hmm_crisis_hit_table.csv
reports/tables/phase_6/india/hmm_crisis_lead_lag_table.csv
reports/tables/phase_6/india/hmm_forward_label_by_state.csv
reports/tables/phase_6/india/hmm_probability_audit.csv
reports/tables/phase_6/india/hmm_no_lookahead_audit.csv
reports/tables/phase_6/india/hmm_metadata.json
```

## 9. Commit vs Local-Only Outputs

Commit:

```text
code
configs
tests
docs
README files
.gitignore
pyproject.toml
.env.example
.gitkeep placeholders
```

Keep local:

```text
data/processed/*hmm*.parquet
models/*hmm*.pkl
reports/tables/phase_6/**/*
reports/figures/phase_6/**/*
logs/*
```

Small summary tables may only be committed later if explicitly approved for final-report release.

## 10. Commands to Regenerate Outputs

Primary plus fallback:

```bash
python scripts/train_regimes.py --market US --model gaussian_hmm --primary --force
python scripts/train_regimes.py --market INDIA --model gaussian_hmm --primary --force
```

Full candidate grid:

```bash
python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force
```

CLI help:

```bash
python scripts/train_regimes.py --help
```

## 11. Tests to Run

Phase-specific tests:

```bash
pytest tests/test_hmm_filtering.py
pytest tests/test_hmm_scaling.py
pytest tests/test_hmm_model.py
pytest tests/test_hmm_no_lookahead.py
```

Shared no-lookahead tests:

```bash
pytest tests/test_no_lookahead.py
```

Full suite:

```bash
pytest
```

## 12. Validation Checklist

A valid Phase 6 run should satisfy:

```text
candidate ranking table exists for US and India
feature availability table exists for US and India
selected model is marked selected_primary=True
filtered probability rows sum to 1
economic probability rows sum to 1
diagnostic smoothed probabilities are clearly named diagnostic-only
no-lookahead audit overall_passed=True
train/test split is chronological
signal trade date is after signal observation date
last row signal_trade_date is missing
threshold/crisis diagnostics are diagnostic-only
HMM panels and model binaries are not tracked by Git
```

## 13. No-Lookahead / Safety Rules

1. HMM features must be point-in-time.
2. `threshold_state`, crisis labels, future labels, and forward/ex-post labels must not enter HMM features.
3. Scaling must be fit on the training window only.
4. HMM parameters must be fit on the training window only.
5. Backtest-facing probabilities must be custom filtered probabilities: `P(S_t | X_1:t)`.
6. Full-sample smoothed probabilities are diagnostic-only.
7. State mapping must use train-period economic properties only.
8. HMM signals computed at date `t` can only be used from the next trading session.
9. Crisis windows are reporting-only.
10. Threshold-regime agreement is reporting-only.

## 14. Known Limitations

1. Gaussian HMM emissions do not directly model autocorrelation in the observed VRP/RV series.
2. Raw HMM states have no inherent economic interpretation and must be mapped.
3. State mapping is sample-sensitive.
4. K=3 can create weak or collapsed states in some feature/covariance combinations.
5. Diagnostic smoothed probabilities are useful for interpretation but cannot be used as tradable signals.
6. Candidate validity depends on occupancy, covariance stability, transition stability, and economic monotonicity.
7. HMM does not establish causal regime structure.

## 15. Review Checklist

Before marking Phase 6 frozen:

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
```

Expected allowed match:

```text
.env.example
```

Optional:

```bash
pytest
```

Reviewer should inspect:

```text
configs/model_hmm.yaml
scripts/train_regimes.py
src/vrp/regimes/gaussian_hmm.py
src/vrp/regimes/online_filter.py
src/vrp/regimes/hmm_registry.py
src/vrp/regimes/hmm_validation.py
src/vrp/regimes/hmm_features.py
src/vrp/regimes/hmm_scaling.py
src/vrp/regimes/state_labeling.py
src/vrp/reports/hmm_diagnostics.py
tests/test_hmm_*.py
tests/test_no_lookahead.py
docs/artifacts/phase_06_artifacts.md
```
