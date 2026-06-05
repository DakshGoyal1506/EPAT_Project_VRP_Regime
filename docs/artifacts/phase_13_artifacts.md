# Phase 13 Artifacts - Cross-Market US-India Analysis

Generated Phase 13 artifacts are local-only by default.

## Artifact Table

| Artifact | Local path | Producer command | Commit? | Reason | Key columns / schema | Review substitute |
|---|---|---|---:|---|---|---|
| Same-date descriptive panel | `data/processed/cross_market_same_date_descriptive_panel.parquet` | `python scripts/run_cross_market_analysis.py --model ALL --force` | No | Generated processed panel | `model`, `panel_type`, `date`, `us_*`, `india_*`, `predictive_allowed` | Shape, schema, head/tail |
| Predictive lagged panel | `data/processed/cross_market_predictive_panel.parquet` | Same | No | Generated processed panel | `model`, `india_date`, `us_lagged_date`, `lag_calendar_days`, lagged US features | Alignment audit and schema preview |
| Combined convenience panel | `data/processed/cross_market_panel.parquet` | Same | No | Generated convenience panel | `panel_type`, descriptive and predictive rows | Separate source panels preferred |
| India overlay panel | `data/processed/india_cross_market_overlay_panel.parquet` | Same | No | Generated overlay diagnostic panel | `date`, `model`, `strategy`, `cutoff`, `base_exposure`, `overlay_exposure`, `analysis_only` | `india_overlay_summary.csv` preview |
| Alignment audit | `reports/tables/phase_13/alignment_audit.csv` | Same | No by default | Critical no-leakage audit | `n_same_date_violations`, lag counts | CSV preview |
| No-lookahead audit | `reports/tables/phase_13/no_lookahead_audit.csv` | Same | No by default | Critical no-lookahead audit | `passes_no_lookahead` | CSV preview |
| VRP level correlations | `reports/tables/phase_13/vrp_level_correlations.csv` | Same | No by default | Diagnostic table | `model`, `pair`, `method`, `correlation`, `p_value` | CSV preview |
| VRP change correlations | `reports/tables/phase_13/vrp_change_correlations.csv` | Same | No by default | Diagnostic table | `model`, `pair`, `method`, `correlation`, `p_value` | CSV preview |
| Regime probability correlations | `reports/tables/phase_13/regime_probability_correlations.csv` | Same | No by default | Diagnostic table | `model`, `pair`, `method`, `correlation`, `p_value` | CSV preview |
| State label agreement | `reports/tables/phase_13/state_label_agreement.csv` | Same | No by default | Diagnostic table | `model`, `table_type`, `us_state_name`, `india_state_name`, `fraction` | CSV preview |
| Lead-lag table | `reports/tables/phase_13/lead_lag_table.csv` | Same | No by default | Diagnostic table | `pair`, `us_lag_trading_rows`, `correlation`, HAC OLS fields | CSV preview |
| Granger diagnostics | `reports/tables/phase_13/granger_diagnostics.csv` | Same | No by default | Descriptive statistical diagnostic | `descriptive_only`, `causal_interpretation_allowed`, `lag`, `p_value` | CSV preview |
| Logistic model summary | `reports/tables/phase_13/logistic_model_summary.csv` | Same | No by default | Predictive diagnostic summary | `model_spec`, `pseudo_r2`, `aic`, `bic`, `auc`, `brier_score` | CSV preview |
| Logistic parameter summary | `reports/tables/phase_13/logistic_parameter_summary.csv` | Same | No by default | Diagnostic coefficient table | `model_spec`, `parameter`, `estimate`, `p_value` | CSV preview |
| Logistic model comparison | `reports/tables/phase_13/logistic_model_comparison.csv` | Same | No by default | Main incremental-US diagnostic | `delta_pseudo_r2`, `delta_aic`, `delta_auc`, `likelihood_ratio_p_value` | CSV preview |
| Logistic OOS diagnostics | `reports/tables/phase_13/logistic_oos_diagnostics.csv` | Same | No by default | Chronological OOS diagnostic | `model_spec`, `auc`, `brier_score`, train/test dates | CSV preview |
| India overlay summary | `reports/tables/phase_13/india_overlay_summary.csv` | Same | No by default | Analysis-only overlay benchmark | `base_sharpe`, `overlay_sharpe`, `blocked_day_fraction`, `analysis_only` | CSV preview |
| Phase 13 metadata | `reports/tables/phase_13/phase13_metadata.json` | Same | No by default | Run metadata and policy record | config snapshot and output paths | JSON preview |
| Phase 13 run status | `reports/tables/phase_13/phase13_run_status.json` | Same | No by default | Run status | `status`, `reason` | JSON preview |
| Summary index | `reports/tables/phase_13/phase13_summary_index.csv` | Same | No by default | Output index | `artifact_name`, `exists_after_write`, status counts | CSV preview |
| VRP figure | `reports/figures/phase_13/us_india_vrp.png` | Same | No by default | Generated diagnostic figure | PNG | Screenshot if needed |
| Stress probability figure | `reports/figures/phase_13/us_india_stress_prob.png` | Same | No by default | Generated diagnostic figure | PNG | Screenshot if needed |
| Lagged US vs India stress figure | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Same | No by default | Generated diagnostic figure | PNG | Screenshot if needed |
| Overlay equity figure | `reports/figures/phase_13/india_overlay_equity_curves.png` | Same | No by default | Generated diagnostic figure | PNG | Screenshot if needed |
| Overlay exposure figure | `reports/figures/phase_13/india_overlay_exposure.png` | Same | No by default | Generated diagnostic figure | PNG | Screenshot if needed |

## Sensitivity / Reproducibility Notes

- Phase 13 artifacts are generated from local upstream Phase 4, 6, 7, 9, and 10 outputs.
- Phase 13 must not read Phase 11 broker or paper-signal artifacts.
- Full parquet panels and generated figures stay local.
- Small excerpts may be copied into the final report only after explicit approval.
- Granger diagnostics are descriptive only and must not be described as causal evidence.
- Overlay artifacts are analysis-only and do not define a new Phase 9 strategy.
