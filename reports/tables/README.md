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

## Phase 7 Examples

Local Phase 7 tables may include:

```text
phase_7/us/mar_metadata.json
phase_7/us/mar_candidate_model_ranking.csv
phase_7/us/mar_state_summary.csv
phase_7/us/mar_transition_matrix.csv
phase_7/us/mar_ar_stability.csv
phase_7/us/mar_probability_audit.csv
phase_7/us/mar_no_lookahead_audit.csv
phase_7/us/mar_duration_summary.csv
phase_7/us/mar_state_by_year.csv
phase_7/us/mar_hmm_agreement.csv
phase_7/us/mar_threshold_agreement.csv
phase_7/india/*.csv
phase_7/india/*.json
regime_model_comparison.csv
```

These are generated run outputs and stay local by default.

Do not commit full generated MAR diagnostic tables unless explicitly selected as final-report artifacts.

## Phase 8 Examples

Local Phase 8 tables may include:

```text
phase_8/us/msvol_metadata.json
phase_8/india/msvol_metadata.json
phase_8/us/msvol_probability_audit.csv
phase_8/india/msvol_probability_audit.csv
phase_8/us/msvol_comparison_summary.csv
phase_8/india/msvol_comparison_summary.csv
phase_8/us/msvol_state_duration_summary.csv
phase_8/india/msvol_state_duration_summary.csv
phase_8/us/msvol_no_lookahead_audit.csv
phase_8/india/msvol_no_lookahead_audit.csv
phase_8/msvol_model_comparison_appendix.csv
phase_8/msvol_no_lookahead_audit.csv
```

These are generated run outputs and stay local by default. Do not commit full generated MSVOL diagnostics unless explicitly selected as final-report artifacts.

## Phase 9 Examples

Local Phase 9 tables may include:

```text
phase_9/strategy_signal_summary.csv
phase_9/strategy_exposure_by_year.csv
phase_9/strategy_exposure_change_summary.csv
phase_9/strategy_blocked_reason_summary.csv
phase_9/strategy_no_lookahead_audit.csv
phase_9/strategy_metadata.json
```

These are generated signal diagnostics and stay local by default.

They must not contain PnL, returns, costs, Sharpe, drawdown, or performance rankings.

## Phase 10 Examples

Local Phase 10 tables may include:

```text
phase_10/backtest_summary.csv
phase_10/backtest_common_start_summary.csv
phase_10/backtest_tail_summary.csv
phase_10/backtest_by_strategy_year.csv
phase_10/crisis_window_performance.csv
phase_10/backtest_availability_summary.csv
phase_10/backtest_no_lookahead_audit.csv
phase_10/backtest_metadata.json
phase_10/robustness_cost_sensitivity.csv
phase_10/robustness_subperiods.csv
phase_10/robustness_weekly_rebalance_skipped.json
phase_10/tradable_proxy_detection.json
phase_10/robustness_metadata.json
phase_10/phase10_final_audit.json
```

These are generated run outputs and stay local by default. Small summaries may be selected later for final report review only if explicitly approved.
