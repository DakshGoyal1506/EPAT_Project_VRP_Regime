# Selected Final-Report Artifacts

This file identifies which artifacts support the final report and how they should be handled.

Generated data panels remain local and are not committed. Small summaries and selected figures may be committed only if explicitly approved for final report review.

## Artifact Selection Rules

1. Prefer summary tables over full panels.
2. Prefer audited CSV/JSON previews over raw data.
3. Prefer selected figures over entire figure directories.
4. Do not commit broker-sensitive artifacts.
5. Do not commit raw data, processed parquet panels, model binaries, or full strategy/backtest/cross-market panels.
6. Every selected table must appear in `reports/final/table_inventory.md`.
7. Every selected figure must appear in `reports/final/figure_inventory.md`.
8. Every major conclusion must appear in `reports/final/result_claims_audit.md`.

## Evidence Inventory

| Report claim area | Primary evidence artifact | Secondary artifact | Commit policy |
|---|---|---|---|
| Data coverage and source validity | `reports/tables/data_audit.csv` | `docs/reproducibility.md`; `configs/data_sources.yaml`; `configs/markets.yaml` | Local-only table; docs/configs tracked |
| Realised variance construction | `reports/tables/rv_summary.csv` | `reports/tables/rv_estimator_correlations.csv`; `reports/figures/rv_estimators_us.png`; `reports/figures/rv_estimators_india.png` | Local-only by default; selected figure optional |
| Implied variance and VRP construction | `reports/tables/vrp_summary.csv` | `reports/tables/vrp_metadata.json`; `reports/figures/us_iv_rv_vrp.png`; `reports/figures/india_iv_rv_vrp.png` | Local-only by default; selected figure optional |
| HAR forecast usefulness | `reports/tables/har_forecast_accuracy.csv` | `reports/tables/har_vrp_summary.csv`; `reports/tables/har_no_lookahead_audit.csv` | Local-only |
| Threshold regime interpretability | `reports/tables/threshold_regime_summary.csv` | `reports/tables/threshold_vrp_by_state.csv`; `reports/tables/threshold_no_lookahead_audit.csv` | Local-only |
| Gaussian HMM regime interpretability | `reports/tables/phase_6/us/hmm_state_summary.csv`; `reports/tables/phase_6/india/hmm_state_summary.csv` | `reports/tables/phase_6/us/hmm_candidate_model_ranking.csv`; `reports/tables/phase_6/india/hmm_candidate_model_ranking.csv`; no-lookahead audits | Local-only |
| MAR as AR-aware regime improvement | `reports/tables/phase_7/us/mar_state_summary.csv`; `reports/tables/phase_7/india/mar_state_summary.csv` | `reports/tables/phase_7/us/mar_ar_stability.csv`; `reports/tables/phase_7/india/mar_ar_stability.csv`; no-lookahead audits | Local-only |
| MSVOL robustness appendix | `reports/tables/phase_8/msvol_model_comparison_appendix.csv` | `reports/tables/phase_8/msvol_no_lookahead_audit.csv` | Local-only |
| Strategy signal design | `reports/tables/phase_9/strategy_signal_summary.csv` | `reports/tables/phase_9/strategy_no_lookahead_audit.csv` | Local-only |
| Phase 10 research-proxy performance | `reports/tables/phase_10/backtest_summary.csv` | `reports/tables/phase_10/backtest_common_start_summary.csv`; `reports/tables/phase_10/backtest_tail_summary.csv`; `reports/figures/phase_10/equity_curves_common_start_us.png`; `reports/figures/phase_10/equity_curves_common_start_india.png` | Local-only by default; selected figures optional |
| Phase 10 drawdown and tail behaviour | `reports/tables/phase_10/backtest_tail_summary.csv` | `reports/tables/phase_10/crisis_window_performance.csv`; `reports/figures/phase_10/drawdowns_us.png`; `reports/figures/phase_10/drawdowns_india.png` | Local-only by default; selected figures optional |
| Robustness and cost sensitivity | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | `reports/tables/phase_10/robustness_subperiods.csv`; `reports/tables/phase_10/robustness_metadata.json` | Local-only |
| Phase 10 no-lookahead and final audit | `reports/tables/phase_10/backtest_no_lookahead_audit.csv` | `reports/tables/phase_10/phase10_final_audit.json` | Local-only |
| Phase 11 paper-signal readiness | `reports/tables/phase_11/live_order_guard_report.json` | `reports/tables/phase_11/risk_check_report.csv`; `reports/tables/phase_11/phase11_integration_report.json` | Local-only; redacted summary only if needed |
| Phase 13 alignment and no-lookahead | `reports/tables/phase_13/alignment_audit.csv` | `reports/tables/phase_13/no_lookahead_audit.csv` | Local-only |
| Phase 13 same-date descriptive co-movement | `reports/tables/phase_13/vrp_level_correlations.csv` | `reports/tables/phase_13/vrp_change_correlations.csv`; `reports/tables/phase_13/regime_probability_correlations.csv`; `reports/tables/phase_13/state_label_agreement.csv`; `reports/figures/phase_13/us_india_vrp.png`; `reports/figures/phase_13/us_india_stress_prob.png` | Local-only by default; selected figures optional |
| Phase 13 lagged-US predictive diagnostics | `reports/tables/phase_13/lead_lag_table.csv` | `reports/tables/phase_13/granger_diagnostics.csv`; `reports/tables/phase_13/logistic_model_comparison.csv`; `reports/tables/phase_13/logistic_oos_diagnostics.csv`; `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Local-only by default; selected figure optional |
| Phase 13 India overlay, analysis-only | `reports/tables/phase_13/india_overlay_summary.csv` | `reports/figures/phase_13/india_overlay_equity_curves.png`; `reports/figures/phase_13/india_overlay_exposure.png` | Local-only by default; selected figure optional |
| Generated artifact policy | `docs/generated_artifact_policy.md` | `docs/artifact_inventory.md`; `.gitignore` | Tracked docs |
| Reproducibility | `docs/reproducibility.md` | `reports/final/reproducibility_note.md`; `docs/commands.md` | Tracked docs |
| Final limitations | `docs/known_limitations.md` | `reports/final/limitations.md` | Tracked docs |
| Final report source | `reports/final/final_report.md` | `reports/final/result_claims_audit.md`; table and figure inventories | Commit |
| Final PDF deliverable | `reports/final/final_report.pdf` | `reports/final/final_report.md`; PDF visual inspection checklist | Commit |

## Selected Candidate Tables for Report Inclusion

Only compact excerpts should be inserted into `final_report.md`.

| Candidate final report table | Source artifact | Inclusion status | Notes |
|---|---|---|---|
| Data coverage summary | `reports/tables/data_audit.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Use market/source/date-range rows only |
| VRP construction summary | `reports/tables/vrp_summary.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | No causal/trading claim |
| HAR forecast summary | `reports/tables/har_forecast_accuracy.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Model-dependent forecast evidence |
| Regime model ladder summary | Phase 5/6/7 summary tables | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Interpretability comparison only |
| Phase 10 strategy summary | `reports/tables/phase_10/backtest_summary.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Research-proxy only |
| Phase 10 robustness summary | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Cost-assumption sensitivity only |
| Phase 11 guard summary | `reports/tables/phase_11/live_order_guard_report.json` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | No broker orders |
| Phase 13 cross-market summary | `reports/tables/phase_13/logistic_model_comparison.csv`; `lead_lag_table.csv` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Predictive/statistical diagnostic only |

## Selected Candidate Figures for Report Inclusion

| Candidate final report figure | Source artifact | Inclusion status | Notes |
|---|---|---|---|
| US/India VRP construction figure | `reports/figures/us_iv_rv_vrp.png`; `reports/figures/india_iv_rv_vrp.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Use if readable |
| US/India common-start proxy curves | `reports/figures/phase_10/equity_curves_common_start_us.png`; `reports/figures/phase_10/equity_curves_common_start_india.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Must label as research-proxy cumulative curves |
| US/India proxy drawdowns | `reports/figures/phase_10/drawdowns_us.png`; `reports/figures/phase_10/drawdowns_india.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Must label as proxy drawdowns |
| US-India VRP comparison | `reports/figures/phase_13/us_india_vrp.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Descriptive only |
| US-India stress probability comparison | `reports/figures/phase_13/us_india_stress_prob.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Descriptive co-movement only |
| Lagged-US vs India stress diagnostic | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Available local; inspected in `local_artifacts/phase14_evidence_review.md` | Predictive diagnostic, not causal proof |
| India overlay exposure | `reports/figures/phase_13/india_overlay_exposure.png` | Available local; optional | Analysis-only overlay |
| India overlay proxy curve | `reports/figures/phase_13/india_overlay_equity_curves.png` | Available local; optional | Analysis-only; not a new strategy |

## Local-Only Artifacts

Keep these local:

```text
data/raw/*
data/interim/*
data/processed/*
data/manual/*
data/broker_cache/*
models/*
logs/*
reports/tables/phase_10/*.csv
reports/tables/phase_10/*.json
reports/tables/phase_11/*.csv
reports/tables/phase_11/*.json
reports/tables/phase_11/*.yaml
reports/tables/phase_13/*.csv
reports/tables/phase_13/*.json
reports/figures/phase_10/*.png
reports/figures/phase_13/*.png
```

## Optional Commit Candidates

Only after explicit approval:

```text
small final summary CSV files
selected final-report figures
redacted validation summaries
```

If selected figures are committed, they should preferably be copied into a final-report-specific path rather than committing entire phase figure directories.

Suggested optional path:

```text
reports/final/figures/
```

This folder now contains the selected final-report figure copies listed below. Do not copy or commit additional figures unless explicitly approved.

## Final Staged Figure Copies

| Final staged artifact | Source artifact | Report section | Commit policy | Caption caveat |
|---|---|---|---|---|
| `reports/final/figures/us_iv_rv_vrp.png` | `reports/figures/us_iv_rv_vrp.png` | IV and VRP construction | Commit | VIX proxy, not variance swap quote |
| `reports/final/figures/india_iv_rv_vrp.png` | `reports/figures/india_iv_rv_vrp.png` | IV and VRP construction | Commit | India VIX proxy, not variance swap quote |
| `reports/final/figures/phase10_equity_curves_common_start_us.png` | `reports/figures/phase_10/equity_curves_common_start_us.png` | Phase 10 backtest | Commit | Research-proxy cumulative curve, not account equity |
| `reports/final/figures/phase10_equity_curves_common_start_india.png` | `reports/figures/phase_10/equity_curves_common_start_india.png` | Phase 10 backtest | Commit | Research-proxy cumulative curve, not account equity |
| `reports/final/figures/phase10_drawdowns_us.png` | `reports/figures/phase_10/drawdowns_us.png` | Phase 10 backtest | Commit | Research-proxy drawdown, not account drawdown |
| `reports/final/figures/phase10_drawdowns_india.png` | `reports/figures/phase_10/drawdowns_india.png` | Phase 10 backtest | Commit | Research-proxy drawdown, not account drawdown |
| `reports/final/figures/phase13_us_india_vrp.png` | `reports/figures/phase_13/us_india_vrp.png` | Phase 13 cross-market | Commit | Descriptive diagnostic, not causal proof |
| `reports/final/figures/phase13_us_india_stress_prob.png` | `reports/figures/phase_13/us_india_stress_prob.png` | Phase 13 cross-market | Commit | Descriptive diagnostic, not causal proof |
| `reports/final/figures/phase13_lagged_us_vs_india_stress.png` | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Phase 13 cross-market | Commit | Predictive diagnostic, not causal proof |

Only the copied files under `reports/final/figures/` are selected for final packaging. The original generated figure directories remain local-only by default.

## Final PDF Inclusion Rule

Every table or figure included in:

```text
reports/final/final_report.pdf
```

must be listed in:

```text
reports/final/table_inventory.md
reports/final/figure_inventory.md
reports/final/selected_artifacts.md
```

Every major conclusion in the PDF must be listed in:

```text
reports/final/result_claims_audit.md
```

## Phase 14 Evidence Inspection Update

The evidence-backed report update pass used `local_artifacts/phase14_evidence_review.md` to inspect local evidence files. Commit policy remains unchanged: generated Phase 10/11/13 CSV, JSON, YAML, and PNG artifacts stay local-only by default.

The selected final-report figure copies are staged under `reports/final/figures/`. This does not approve table commits and does not approve committing original generated figure directories.
