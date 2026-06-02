# Report Tables

Generated diagnostic and summary tables are written here.

## Commit Policy

By default, generated tables stay local.

Commit only:

```text
README.md
.gitkeep
small final-report-ready tables explicitly approved for review
```

Do not commit by default:

```text
*.csv
*.json
*.parquet
*.xlsx
```

## Examples

Local generated tables may include:

```text
data_audit.csv
vrp_summary.csv
har_forecast_accuracy.csv
har_coefficients.csv
har_vrp_summary.csv
har_metadata.json
har_no_lookahead_audit.csv
backtest_summary.csv
phase_10/*.csv
phase_11/*.csv
```

Use `docs/artifact_inventory.md` to document which local table should be sent as a review substitute.

## Phase 5 Examples

Local Phase 5 tables may include:

```text
threshold_regime_summary.csv
threshold_component_summary.csv
threshold_transition_matrix.csv
threshold_state_duration_summary.csv
threshold_state_by_year.csv
threshold_crisis_hit_table.csv
threshold_crisis_lead_lag_table.csv
threshold_vrp_by_state.csv
threshold_forward_label_by_state.csv
threshold_no_lookahead_audit.csv
threshold_regime_metadata.json
```

These are generated run outputs and stay local by default.

## Phase 6 Examples

Local Phase 6 tables may include:

```text
phase_6/us/hmm_candidate_model_ranking.csv
phase_6/us/hmm_feature_availability.csv
phase_6/us/hmm_probability_audit.csv
phase_6/us/hmm_no_lookahead_audit.csv
phase_6/us/hmm_metadata.json
phase_6/india/hmm_candidate_model_ranking.csv
phase_6/india/hmm_feature_availability.csv
phase_6/india/hmm_probability_audit.csv
phase_6/india/hmm_no_lookahead_audit.csv
phase_6/india/hmm_metadata.json
```

These are generated run outputs and stay local by default.
