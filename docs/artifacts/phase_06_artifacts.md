# Phase 6 Artifacts - Gaussian HMM Regime Model

## Commit Policy

Phase 6 generated artifacts are local-only by default.

Commit:

```text
code
configs
tests
docs
README files
.gitkeep placeholders
```

Do not commit:

```text
data/processed/*hmm*.parquet
models/*hmm*.pkl
reports/tables/phase_6/**/*
reports/figures/phase_6/**/*
logs/*
```

## Artifact Table

| Artifact name | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute | Notes |
|---|---|---|---:|---|---|---|---|
| US selected HMM regime panel | `data/processed/us_hmm_regimes.parquet` | `python scripts/train_regimes.py --market US --model gaussian_hmm --primary --force` | No | Generated model output | `date`, HMM states, filtered probabilities, t+1 signal columns | schema/head/tail preview, no-lookahead audit | Local regenerated panel |
| India selected HMM regime panel | `data/processed/india_hmm_regimes.parquet` | `python scripts/train_regimes.py --market INDIA --model gaussian_hmm --primary --force` | No | Generated model output | `date`, HMM states, filtered probabilities, t+1 signal columns | schema/head/tail preview, no-lookahead audit | Local regenerated panel |
| HMM model-specific panels | `data/processed/*_hmm_*.parquet` | `python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force` | No | Generated candidate panels | candidate-specific HMM state/probability columns | candidate ranking table | One file per market/feature/K/covariance candidate |
| US selected HMM model bundle | `models/us_gaussian_hmm.pkl` | Same | No | Serialized model/scaler/state-label bundle | joblib/pickle bundle | metadata JSON, config, candidate ranking | Local binary |
| India selected HMM model bundle | `models/india_gaussian_hmm.pkl` | Same | No | Serialized model/scaler/state-label bundle | joblib/pickle bundle | metadata JSON, config, candidate ranking | Local binary |
| HMM model-specific bundles | `models/*_hmm_*.pkl` | Same | No | Generated model binaries | model, scaler, metadata, state mapping | candidate ranking, metadata | Local binary |
| Candidate ranking | `reports/tables/phase_6/{market}/hmm_candidate_model_ranking.csv` | Same | No by default | Generated diagnostics | `market`, `feature_set`, `n_states`, `covariance_type`, `aic`, `bic`, `selected_primary`, `rejection_reason` | CSV preview | Small but generated |
| Feature availability | `reports/tables/phase_6/{market}/hmm_feature_availability.csv` | Same | No by default | Generated diagnostics | `required_feature`, `required_condition`, `n_missing_feature`, `n_condition_failed` | CSV preview | Confirms HAR availability gating |
| State summary | `reports/tables/phase_6/{market}/hmm_state_summary.csv` | Same | No by default | Generated diagnostics | state name, occupancy, mean probabilities, mean features | CSV preview | Used for interpretation |
| Transition matrix | `reports/tables/phase_6/{market}/hmm_transition_matrix.csv` | Same | No by default | Generated diagnostics | from/to raw state, economic labels, transition probability | CSV preview | Fitted HMM transition diagnostics |
| State duration summary | `reports/tables/phase_6/{market}/hmm_state_duration_summary.csv` | Same | No by default | Generated diagnostics | state, run counts, duration stats | CSV preview | Persistence diagnostics |
| State by year | `reports/tables/phase_6/{market}/hmm_state_by_year.csv` | Same | No by default | Generated diagnostics | year, state, annual occupancy | CSV preview | Annual regime distribution |
| Threshold agreement | `reports/tables/phase_6/{market}/hmm_threshold_agreement.csv` | Same | No by default | Diagnostic comparison | HMM state, threshold state, agreement count/rate | CSV preview | Threshold states are not HMM inputs |
| Crisis hit table | `reports/tables/phase_6/{market}/hmm_crisis_hit_table.csv` | Same | No by default | Diagnostic comparison | crisis window, stress overlap, false negatives | CSV preview | Crisis windows are not training labels |
| Crisis lead-lag table | `reports/tables/phase_6/{market}/hmm_crisis_lead_lag_table.csv` | Same | No by default | Diagnostic comparison | crisis window, first stress date, lead/lag | CSV preview | Diagnostic-only |
| Forward-label by state | `reports/tables/phase_6/{market}/hmm_forward_label_by_state.csv` | Same | No by default | Diagnostic comparison | forward label, state, frequency | CSV preview | Forward labels are not HMM inputs |
| Probability audit | `reports/tables/phase_6/{market}/hmm_probability_audit.csv` | Same | No by default | No-lookahead/probability audit | filtered/smoothed/backtest columns, row sums, prefix invariance | CSV preview | Critical review substitute |
| No-lookahead audit | `reports/tables/phase_6/{market}/hmm_no_lookahead_audit.csv` | Same | No by default | Timing and leakage audit | check name, passed, details, overall_passed | CSV preview | Critical review substitute |
| Metadata | `reports/tables/phase_6/{market}/hmm_metadata.json` | Same | No by default | Run reproducibility metadata | hashes, selected model, candidates, config hash | JSON preview | Local run provenance |
| Figures | `reports/figures/phase_6/*` | Future Phase 6 figure generation if added | No by default | Generated figures | figure-specific | screenshot | No required Phase 6 figures currently |

## Key Output Columns for HMM Panels

Backtest-facing regime columns:

```text
hmm_signal_observation_date
hmm_signal_available_after_close_date
hmm_signal_trade_date
hmm_state_for_next_session
hmm_state_name_for_next_session
hmm_filtered_prob_calm_for_next_session
hmm_filtered_prob_transition_for_next_session
hmm_filtered_prob_stress_for_next_session
```

Filtered probability columns:

```text
hmm_filtered_prob_raw_state_0
hmm_filtered_prob_raw_state_1
hmm_filtered_prob_raw_state_2
hmm_filtered_prob_calm
hmm_filtered_prob_transition
hmm_filtered_prob_stress
```

Diagnostic-only smoothed columns:

```text
hmm_diagnostic_smoothed_prob_raw_state_0
hmm_diagnostic_smoothed_prob_raw_state_1
hmm_diagnostic_smoothed_prob_raw_state_2
```

## Review Substitutes for Large Artifacts

Use these instead of committing generated panels or model binaries:

```text
pytest output
hmm_candidate_model_ranking.csv preview
hmm_probability_audit.csv preview
hmm_no_lookahead_audit.csv preview
hmm_metadata.json preview
schema/head/tail printout for selected HMM panels
```

## Sensitivity / Reproducibility Notes

1. Model binaries are generated and local-only.
2. HMM outputs can change with dependency versions, random seed, feature availability, and data refresh.
3. Config, hashes, metadata, and candidate ranking must be used to document exact run context.
4. Broker data is not involved in Phase 6.
5. Smoothed probabilities are diagnostic-only and must not be used for tradable decisions.
