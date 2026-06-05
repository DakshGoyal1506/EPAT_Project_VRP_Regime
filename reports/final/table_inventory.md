# Final Report Table Inventory

This file inventories every table used or referenced in the final report and PDF.

Do not include a table in `final_report.md` or `final_report.pdf` unless it appears here.

## Table Status Legend

| Status | Meaning |
|---|---|
| `available-local` | File exists locally but is not tracked by Git |
| `tracked` | File is committed to Git |
| `placeholder` | Expected artifact not inspected yet |
| `not-used` | Candidate artifact exists but is not selected for the final report |
| `to-create` | Final-report table to be manually written or derived from audited artifacts |

## Commit Policy Legend

| Policy | Meaning |
|---|---|
| `commit` | Commit the final report table or Markdown document |
| `local-only` | Keep generated artifact local |
| `optional-selected` | Commit only if explicitly approved as a small final-report artifact |
| `do-not-commit` | Never commit this artifact |

## Inventory

| Table ID | Report section | Source artifact path | Producer command | Status | Commit policy | Notes |
|---|---|---|---|---|---|---|
| T00 | Appendix / claims audit | `reports/final/result_claims_audit.md` | Manual Phase 14 | to-create | commit | Mandatory claim-to-evidence audit |
| T01 | Data | `reports/tables/data_audit.csv` | `python scripts/download_data.py ...` | placeholder | local-only | Use only if local file is available and inspected |
| T02 | Realised variance construction | `reports/tables/rv_summary.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | placeholder | local-only | Supports RV estimator summary |
| T03 | Realised variance construction | `reports/tables/rv_estimator_correlations.csv` | `python scripts/build_features.py --market ALL --feature rv --window 22` | placeholder | local-only | Optional estimator comparison |
| T04 | Implied variance and VRP construction | `reports/tables/vrp_summary.csv` | `python scripts/build_features.py --market ALL --feature vrp` | placeholder | local-only | Supports VRP construction summary |
| T05 | Implied variance and VRP construction | `reports/tables/vrp_metadata.json` | `python scripts/build_features.py --market ALL --feature vrp` | placeholder | local-only | Supports horizon/source/formula notes |
| T06 | HAR-RV forecasting | `reports/tables/har_forecast_accuracy.csv` | `python scripts/train_har.py --market ALL --mode expanding ...` | placeholder | local-only | Supports HAR forecast evaluation |
| T07 | HAR-RV forecasting | `reports/tables/har_vrp_summary.csv` | `python scripts/train_har.py --market ALL --mode expanding ...` | placeholder | local-only | Supports HAR-based VRP summary |
| T08 | HAR-RV forecasting | `reports/tables/har_no_lookahead_audit.csv` | `python scripts/train_har.py --market ALL --mode expanding ...` | placeholder | local-only | Supports point-in-time forecast caveat |
| T09 | Threshold regimes | `reports/tables/threshold_regime_summary.csv` | `python scripts/train_regimes.py --model threshold --market ALL --force` | placeholder | local-only | Supports threshold regime distribution |
| T10 | Threshold regimes | `reports/tables/threshold_vrp_by_state.csv` | `python scripts/train_regimes.py --model threshold --market ALL --force` | placeholder | local-only | Supports VRP-by-regime narrative |
| T11 | Threshold regimes | `reports/tables/threshold_no_lookahead_audit.csv` | `python scripts/train_regimes.py --model threshold --market ALL --force` | placeholder | local-only | Supports threshold no-lookahead statement |
| T12 | Gaussian HMM | `reports/tables/phase_6/us/hmm_candidate_model_ranking.csv` | `python scripts/train_regimes.py --market ALL --model gaussian_hmm --run-grid --force` | placeholder | local-only | US HMM model-selection evidence |
| T13 | Gaussian HMM | `reports/tables/phase_6/india/hmm_candidate_model_ranking.csv` | Same as T12 | placeholder | local-only | India HMM model-selection evidence |
| T14 | Gaussian HMM | `reports/tables/phase_6/us/hmm_state_summary.csv` | Same as T12 | placeholder | local-only | US HMM state interpretation |
| T15 | Gaussian HMM | `reports/tables/phase_6/india/hmm_state_summary.csv` | Same as T12 | placeholder | local-only | India HMM state interpretation |
| T16 | Gaussian HMM | `reports/tables/phase_6/us/hmm_no_lookahead_audit.csv` | Same as T12 | placeholder | local-only | US HMM no-lookahead evidence |
| T17 | Gaussian HMM | `reports/tables/phase_6/india/hmm_no_lookahead_audit.csv` | Same as T12 | placeholder | local-only | India HMM no-lookahead evidence |
| T18 | Markov autoregression | `reports/tables/phase_7/us/mar_candidate_model_ranking.csv` | `python scripts/train_markov_autoreg.py --market ALL --target vrp_har --order 1 --states 2 --primary --force` | placeholder | local-only | US MAR candidate evidence |
| T19 | Markov autoregression | `reports/tables/phase_7/india/mar_candidate_model_ranking.csv` | Same as T18 | placeholder | local-only | India MAR candidate evidence |
| T20 | Markov autoregression | `reports/tables/phase_7/us/mar_state_summary.csv` | Same as T18 | placeholder | local-only | US MAR state interpretation |
| T21 | Markov autoregression | `reports/tables/phase_7/india/mar_state_summary.csv` | Same as T18 | placeholder | local-only | India MAR state interpretation |
| T22 | Markov autoregression | `reports/tables/phase_7/us/mar_ar_stability.csv` | Same as T18 | placeholder | local-only | US AR stability evidence |
| T23 | Markov autoregression | `reports/tables/phase_7/india/mar_ar_stability.csv` | Same as T18 | placeholder | local-only | India AR stability evidence |
| T24 | Markov autoregression | `reports/tables/phase_7/us/mar_no_lookahead_audit.csv` | Same as T18 | placeholder | local-only | US MAR no-lookahead evidence |
| T25 | Markov autoregression | `reports/tables/phase_7/india/mar_no_lookahead_audit.csv` | Same as T18 | placeholder | local-only | India MAR no-lookahead evidence |
| T26 | MSVOL appendix | `reports/tables/phase_8/msvol_model_comparison_appendix.csv` | `python scripts/run_msvol_diagnostics.py --market ALL` | placeholder | local-only | Python-only MSVOL robustness evidence |
| T27 | MSVOL appendix | `reports/tables/phase_8/msvol_no_lookahead_audit.csv` | `python scripts/run_msvol_no_lookahead_audit.py --market ALL` | placeholder | local-only | MSVOL audit evidence |
| T28 | Strategy construction | `reports/tables/phase_9/strategy_signal_summary.csv` | `python scripts/build_signals.py --market ALL --strategy all --force` | placeholder | local-only | Signal availability/exposure evidence |
| T29 | Strategy construction | `reports/tables/phase_9/strategy_no_lookahead_audit.csv` | Same as T28 | placeholder | local-only | Strategy signal timing evidence |
| T30 | Vectorised research backtest | `reports/tables/phase_10/backtest_summary.csv` | `python scripts/run_backtest.py --market ALL --strategy all --cost-bps 5 --force` | available-local | local-only | Key Phase 10 research-proxy performance table |
| T31 | Vectorised research backtest | `reports/tables/phase_10/backtest_common_start_summary.csv` | `python scripts/generate_backtest_diagnostics.py --market ALL` | available-local | local-only | Common-start comparison table |
| T32 | Vectorised research backtest | `reports/tables/phase_10/backtest_tail_summary.csv` | Same as T31 | available-local | local-only | Tail/drawdown evidence |
| T33 | Vectorised research backtest | `reports/tables/phase_10/backtest_by_strategy_year.csv` | Same as T31 | available-local | local-only | Year-level diagnostics |
| T34 | Vectorised research backtest | `reports/tables/phase_10/crisis_window_performance.csv` | Same as T31 | available-local | local-only | Crisis-window diagnostics |
| T35 | Vectorised research backtest | `reports/tables/phase_10/backtest_availability_summary.csv` | Same as T31 | available-local | local-only | Sample availability evidence |
| T36 | Vectorised research backtest | `reports/tables/phase_10/backtest_no_lookahead_audit.csv` | Same as T31 | available-local | local-only | Backtest timing audit |
| T37 | Vectorised research backtest | `reports/tables/phase_10/phase10_final_audit.json` | `python scripts/audit_phase10_final.py --market ALL` | available-local | local-only | Final Phase 10 audit |
| T38 | Robustness checks | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | `python scripts/run_robustness.py --market ALL --test all --force` | available-local | local-only | Cost sensitivity evidence |
| T39 | Robustness checks | `reports/tables/phase_10/robustness_subperiods.csv` | Same as T38 | available-local | local-only | Subperiod evidence |
| T40 | Robustness checks | `reports/tables/phase_10/tradable_proxy_detection.json` | Same as T38 | available-local | local-only | Confirms proxy nature / tradable proxy detection |
| T41 | Phase 11 paper-signal readiness | `reports/tables/phase_11/risk_check_report.csv` | `python scripts/run_ibkr_paper_signal.py --market US --strategy mar_prob_linear_carry --print-json` | available-local | local-only | Paper-signal risk check |
| T42 | Phase 11 paper-signal readiness | `reports/tables/phase_11/phase11_integration_report.json` | `python scripts/validate_phase11.py --print-json` | available-local | local-only | Phase 11 integration validation |
| T43 | Phase 11 paper-signal readiness | `reports/tables/phase_11/live_order_guard_report.json` | `python scripts/validate_phase11.py --print-json` | available-local | local-only | Live-order guard evidence |
| T44 | Cross-market analysis | `reports/tables/phase_13/alignment_audit.csv` | `python scripts/run_cross_market_analysis.py --model ALL --force` | available-local | local-only | Alignment evidence |
| T45 | Cross-market analysis | `reports/tables/phase_13/no_lookahead_audit.csv` | Same as T44 | available-local | local-only | Cross-market no-lookahead evidence |
| T46 | Cross-market analysis | `reports/tables/phase_13/vrp_level_correlations.csv` | Same as T44 | available-local | local-only | Same-date descriptive correlation |
| T47 | Cross-market analysis | `reports/tables/phase_13/vrp_change_correlations.csv` | Same as T44 | available-local | local-only | Change correlation diagnostics |
| T48 | Cross-market analysis | `reports/tables/phase_13/regime_probability_correlations.csv` | Same as T44 | available-local | local-only | Regime probability co-movement |
| T49 | Cross-market analysis | `reports/tables/phase_13/state_label_agreement.csv` | Same as T44 | available-local | local-only | Regime label agreement |
| T50 | Cross-market analysis | `reports/tables/phase_13/lead_lag_table.csv` | Same as T44 | available-local | local-only | Lead-lag diagnostic |
| T51 | Cross-market analysis | `reports/tables/phase_13/granger_diagnostics.csv` | Same as T44 | available-local | local-only | Granger-style diagnostic; not causal proof |
| T52 | Cross-market analysis | `reports/tables/phase_13/logistic_model_comparison.csv` | Same as T44 | available-local | local-only | Predictive diagnostic |
| T53 | Cross-market analysis | `reports/tables/phase_13/logistic_oos_diagnostics.csv` | Same as T44 | available-local | local-only | Out-of-sample logistic diagnostics |
| T54 | Cross-market analysis | `reports/tables/phase_13/india_overlay_summary.csv` | Same as T44 | available-local | local-only | Analysis-only overlay evidence |
| T55 | Cross-market analysis | `reports/tables/phase_13/phase13_summary_index.csv` | Same as T44 | available-local | local-only | Summary index for Phase 13 artifacts |
| T56 | Reproducibility | `docs/reproducibility.md` | Manual docs | tracked | commit | Reviewer reproducibility workflow |
| T57 | Limitations | `docs/known_limitations.md` | Manual docs | tracked | commit | Source for curated final limitations |
| T58 | Final limitations | `reports/final/limitations.md` | Manual Phase 14 | to-create | commit | Curated final report caveats |
| T59 | Final selected artifacts | `reports/final/selected_artifacts.md` | Manual Phase 14 | to-create | commit | Evidence inventory |
| T60 | Final report checklist | `docs/final_report_checklist.md` | Manual Phase 14 | to-create | commit | Final report QA |

## Tables Selected for Final Report Body

These tables are candidates for concise inclusion in `final_report.md`. Values must be pasted only after inspection.

| Final table | Source table | Inclusion rule |
|---|---|---|
| Data coverage summary | `reports/tables/data_audit.csv` | Include only compact market-level rows |
| VRP summary | `reports/tables/vrp_summary.csv` | Include selected summary metrics only |
| HAR forecast summary | `reports/tables/har_forecast_accuracy.csv` | Include compact forecast comparison only |
| Regime model ladder summary | Phase 5/6/7 summary tables | Include qualitative/compact metrics only |
| Phase 10 strategy summary | `reports/tables/phase_10/backtest_summary.csv` | Include selected strategy rows only after claims audit |
| Phase 10 robustness summary | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | Include compact sensitivity summary only |
| Phase 11 readiness summary | `reports/tables/phase_11/live_order_guard_report.json`; `risk_check_report.csv` | Include guard status only; no broker-sensitive fields |
| Phase 13 cross-market summary | `reports/tables/phase_13/logistic_model_comparison.csv`; `lead_lag_table.csv`; `india_overlay_summary.csv` | Include only statistical-diagnostic wording |

## Tables Not to Include Directly

Do not paste huge or sensitive tables into the report:

```text
full backtest panels
full strategy signal panels
full regime probability panels
full cross-market panels
broker metadata with account or connection details
raw source data
processed parquet contents
large per-date tables
```
