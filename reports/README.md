# Reports

This directory stores final report deliverables and locally generated report artifacts.

Most report outputs are generated locally and are intentionally not tracked by Git. The Phase 14 final report package is the explicit release exception.

## Directory Contract

| Path               | Purpose                                                                                                                                                               |       Commit? |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------: |
| `reports/final/`   | Final report package: Markdown source, PDF export, executive summary, presentation outline, claims audit, inventories, limitations, reproducibility note, future work |           Yes |
| `reports/tables/`  | Generated CSV/JSON/Markdown diagnostic tables                                                                                                                         | No by default |
| `reports/figures/` | Generated figures                                                                                                                                                     | No by default |

## Commit Policy

Commit:

```text
reports/README.md
reports/tables/README.md
reports/tables/.gitkeep
reports/figures/README.md
reports/figures/.gitkeep
reports/final/*.md
reports/final/final_report.pdf
selected final-report-ready artifacts only if explicitly approved
```

Do not commit by default:

```text
reports/tables/*.csv
reports/tables/*.json
reports/figures/*.png
reports/figures/*.svg
reports/figures/*.pdf
```

## Final Report Package

The final report source is:

```text
reports/final/final_report.md
```

The PDF deliverable is:

```text
reports/final/final_report.pdf
```

The PDF must be generated from the Markdown source after claims audit completion. Do not manually diverge the PDF from the Markdown source.

Every major report claim must appear in:

```text
reports/final/result_claims_audit.md
```

## Rules

1. Generated tables must be reproducible from scripts.
2. Generated figures must be reproducible from scripts or report modules.
3. Do not manually edit generated artifacts without documenting the change.
4. Final report conclusions must trace back to reproducible outputs.
5. Broker-sensitive outputs must remain local or be redacted before review.

## Phase 4 HAR-RV Outputs

Phase 4 may generate:

```text
reports/tables/har_forecast_accuracy.csv
reports/tables/har_coefficients.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_metadata.json
reports/tables/har_no_lookahead_audit.csv
reports/figures/har_forecast_us.png
reports/figures/har_forecast_india.png
reports/figures/har_residuals_us.png
reports/figures/har_residuals_india.png
reports/figures/har_vrp_us.png
reports/figures/har_vrp_india.png
```

These are local-only by default. Use previews, selected excerpts, or screenshots for review unless explicitly approved as final-report artifacts.

## Phase 5 Threshold-Regime Outputs

Phase 5 may generate:

```text
reports/tables/threshold_regime_summary.csv
reports/tables/threshold_component_summary.csv
reports/tables/threshold_transition_matrix.csv
reports/tables/threshold_state_duration_summary.csv
reports/tables/threshold_state_by_year.csv
reports/tables/threshold_crisis_hit_table.csv
reports/tables/threshold_crisis_lead_lag_table.csv
reports/tables/threshold_vrp_by_state.csv
reports/tables/threshold_forward_label_by_state.csv
reports/tables/threshold_no_lookahead_audit.csv
reports/tables/threshold_regime_metadata.json
reports/figures/threshold_regimes_us.png
reports/figures/threshold_regimes_india.png
reports/figures/threshold_regime_vrp_boxplots_us.png
reports/figures/threshold_regime_vrp_boxplots_india.png
reports/figures/threshold_component_states_us.png
reports/figures/threshold_component_states_india.png
```

These are generated diagnostics and stay local by default. Use previews, selected excerpts, or screenshots for review unless explicitly approved as final-report artifacts.

## Phase 6 Gaussian HMM Outputs

Phase 6 may generate:

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

reports/tables/phase_6/india/*.csv
reports/tables/phase_6/india/*.json
reports/figures/phase_6/*
```

These are generated diagnostics and stay local by default. Use selected CSV/JSON previews for review unless explicitly approved as final-report artifacts.

## Phase 7 Markov Autoregression Outputs

Phase 7 may generate:

```text
reports/tables/phase_7/us/mar_metadata.json
reports/tables/phase_7/us/mar_candidate_model_ranking.csv
reports/tables/phase_7/us/mar_state_summary.csv
reports/tables/phase_7/us/mar_transition_matrix.csv
reports/tables/phase_7/us/mar_ar_stability.csv
reports/tables/phase_7/us/mar_probability_audit.csv
reports/tables/phase_7/us/mar_no_lookahead_audit.csv
reports/tables/phase_7/us/mar_duration_summary.csv
reports/tables/phase_7/us/mar_state_by_year.csv
reports/tables/phase_7/us/mar_hmm_agreement.csv
reports/tables/phase_7/us/mar_threshold_agreement.csv
reports/tables/phase_7/india/*.csv
reports/tables/phase_7/india/*.json
reports/tables/regime_model_comparison.csv
reports/figures/phase_7/*
```

These are generated diagnostics and stay local by default. Use selected CSV/JSON previews, terminal output, or screenshots for review unless explicitly approved as final-report artifacts.

Filtered probabilities are backtest-facing. Smoothed probabilities are diagnostic-only.

## Phase 8 MSVOL Robustness Appendix Outputs

Phase 8 may generate:

```text
reports/tables/phase_8/us/msvol_metadata.json
reports/tables/phase_8/india/msvol_metadata.json
reports/tables/phase_8/us/msvol_probability_audit.csv
reports/tables/phase_8/india/msvol_probability_audit.csv
reports/tables/phase_8/us/msvol_comparison_summary.csv
reports/tables/phase_8/india/msvol_comparison_summary.csv
reports/tables/phase_8/us/msvol_state_duration_summary.csv
reports/tables/phase_8/india/msvol_state_duration_summary.csv
reports/tables/phase_8/us/msvol_no_lookahead_audit.csv
reports/tables/phase_8/india/msvol_no_lookahead_audit.csv
reports/tables/phase_8/msvol_model_comparison_appendix.csv
reports/tables/phase_8/msvol_no_lookahead_audit.csv
reports/figures/phase_8/*
```

These are generated diagnostics and stay local by default. Phase 8 is Python-only MSVOL, not true MSGARCH. It is diagnostic-only and is not used for strategy construction or backtesting.

## Phase 9 Strategy Signal Outputs

Phase 9 may generate:

```text
reports/tables/phase_9/strategy_signal_summary.csv
reports/tables/phase_9/strategy_exposure_by_year.csv
reports/tables/phase_9/strategy_exposure_change_summary.csv
reports/tables/phase_9/strategy_blocked_reason_summary.csv
reports/tables/phase_9/strategy_no_lookahead_audit.csv
reports/tables/phase_9/strategy_metadata.json
reports/figures/phase_9/*
```

These are generated signal diagnostics and stay local by default. They do not contain PnL, returns, costs, Sharpe, drawdown, or performance rankings. Use selected CSV/JSON previews for review unless explicitly approved as final-report artifacts.

## Phase 10 Vectorised Backtest Outputs

Phase 10 may generate:

```text
reports/tables/phase_10/backtest_summary.csv
reports/tables/phase_10/backtest_common_start_summary.csv
reports/tables/phase_10/backtest_tail_summary.csv
reports/tables/phase_10/backtest_by_strategy_year.csv
reports/tables/phase_10/crisis_window_performance.csv
reports/tables/phase_10/backtest_availability_summary.csv
reports/tables/phase_10/backtest_no_lookahead_audit.csv
reports/tables/phase_10/backtest_metadata.json
reports/tables/phase_10/robustness_cost_sensitivity.csv
reports/tables/phase_10/robustness_subperiods.csv
reports/tables/phase_10/robustness_weekly_rebalance_skipped.json
reports/tables/phase_10/tradable_proxy_detection.json
reports/tables/phase_10/robustness_metadata.json
reports/tables/phase_10/phase10_final_audit.json
reports/figures/phase_10/equity_curves_*.png
reports/figures/phase_10/equity_curves_common_start_*.png
reports/figures/phase_10/drawdowns_*.png
reports/figures/phase_10/return_distribution_*.png
```

These are generated diagnostics and stay local by default. The cumulative curves are additive research proxy sums over overlapping forward labels, not executable account equity curves.

## Phase 11 Broker Readiness Outputs

Phase 11 may generate:

```text
reports/tables/phase_11/daily_paper_signal.csv
reports/tables/phase_11/paper_order_intents.csv
reports/tables/phase_11/risk_check_report.csv
reports/tables/phase_11/broker_metadata.json
reports/tables/phase_11/run_metadata.json
reports/tables/phase_11/ibkr_paper_config_snapshot.yaml
reports/tables/phase_11/phase11_integration_report.json
reports/tables/phase_11/live_order_guard_report.json
```

These are runtime broker-readiness artifacts and stay local by default. Commit only README files or explicitly approved redacted summaries.

## Phase 13 Cross-Market Outputs

Phase 13 may generate:

```text
reports/tables/phase_13/alignment_audit.csv
reports/tables/phase_13/no_lookahead_audit.csv
reports/tables/phase_13/vrp_level_correlations.csv
reports/tables/phase_13/vrp_change_correlations.csv
reports/tables/phase_13/regime_probability_correlations.csv
reports/tables/phase_13/state_label_agreement.csv
reports/tables/phase_13/lead_lag_table.csv
reports/tables/phase_13/granger_diagnostics.csv
reports/tables/phase_13/logistic_model_summary.csv
reports/tables/phase_13/logistic_parameter_summary.csv
reports/tables/phase_13/logistic_model_comparison.csv
reports/tables/phase_13/logistic_oos_diagnostics.csv
reports/tables/phase_13/india_overlay_summary.csv
reports/tables/phase_13/phase13_summary_index.csv
reports/tables/phase_13/phase13_metadata.json
reports/tables/phase_13/phase13_run_status.json
reports/figures/phase_13/*.png
```

These are generated diagnostics and stay local by default. Same-date cross-market diagnostics are descriptive only. Predictive diagnostics must use lagged US information only. The overlay is analysis-only and is not a new Phase 9 strategy.
