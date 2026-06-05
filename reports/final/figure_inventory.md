# Final Report Figure Inventory

This file inventories every figure used or referenced in the final report and PDF.

Do not include a figure in `final_report.md` or `final_report.pdf` unless it appears here.

## Figure Status Legend

| Status | Meaning |
|---|---|
| `available-local` | File exists locally but is not tracked by Git |
| `tracked` | File is committed to Git |
| `placeholder` | Expected artifact not inspected yet |
| `not-used` | Candidate figure exists but is not selected for the final report |
| `to-create` | Final-report figure to be created or exported later |
| `staged-final` | Selected final-package copy staged under `reports/final/figures/` |

## Commit Policy Legend

| Policy | Meaning |
|---|---|
| `commit` | Commit the final figure or Markdown document |
| `local-only` | Keep generated figure local |
| `optional-selected` | Commit only if explicitly approved as a final-report figure |
| `do-not-commit` | Never commit this artifact |

## Inventory

| Figure ID | Report section | Source artifact path | Producer command | Status | Commit policy | Notes |
|---|---|---|---|---|---|---|
| F01 | Realised variance construction | `reports/figures/rv_estimators_us.png` | `python scripts/build_features.py --market ALL --feature rv --window 22` | available-local | optional-selected | US daily OHLC realised-variance estimator diagnostic. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F02 | Realised variance construction | `reports/figures/rv_estimators_india.png` | Same as F01 | available-local | optional-selected | India daily OHLC realised-variance estimator diagnostic. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F03 | Implied variance and VRP construction | `reports/figures/us_iv_rv_vrp.png` | `python scripts/build_features.py --market ALL --feature vrp` | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/us_iv_rv_vrp.png`. |
| F04 | Implied variance and VRP construction | `reports/figures/india_iv_rv_vrp.png` | Same as F03 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/india_iv_rv_vrp.png`. |
| F05 | HAR-RV forecasting | `reports/figures/har_forecast_us.png` | `python scripts/train_har.py --market ALL --mode expanding ...` | placeholder | optional-selected | US HAR forecast diagnostic |
| F06 | HAR-RV forecasting | `reports/figures/har_forecast_india.png` | Same as F05 | placeholder | optional-selected | India HAR forecast diagnostic |
| F07 | HAR-RV forecasting | `reports/figures/har_vrp_us.png` | Same as F05 | placeholder | optional-selected | US HAR-VRP diagnostic |
| F08 | HAR-RV forecasting | `reports/figures/har_vrp_india.png` | Same as F05 | placeholder | optional-selected | India HAR-VRP diagnostic |
| F09 | Threshold regimes | `reports/figures/threshold_regimes_us.png` | `python scripts/train_regimes.py --model threshold --market ALL --force` | placeholder | optional-selected | US threshold regimes |
| F10 | Threshold regimes | `reports/figures/threshold_regimes_india.png` | Same as F09 | placeholder | optional-selected | India threshold regimes |
| F11 | Threshold regimes | `reports/figures/threshold_regime_vrp_boxplots_us.png` | Same as F09 | placeholder | optional-selected | US VRP by threshold state |
| F12 | Threshold regimes | `reports/figures/threshold_regime_vrp_boxplots_india.png` | Same as F09 | placeholder | optional-selected | India VRP by threshold state |
| F13 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_us.png` | `python scripts/generate_backtest_diagnostics.py --market ALL` | available-local | optional-selected | Additive research-proxy curve, not account equity |
| F14 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_india.png` | Same as F13 | available-local | optional-selected | Additive research-proxy curve, not account equity |
| F15 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_common_start_us.png` | Same as F13 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase10_equity_curves_common_start_us.png`. |
| F16 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_common_start_india.png` | Same as F13 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase10_equity_curves_common_start_india.png`. |
| F17 | Vectorised research backtest | `reports/figures/phase_10/drawdowns_us.png` | Same as F13 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase10_drawdowns_us.png`. |
| F18 | Vectorised research backtest | `reports/figures/phase_10/drawdowns_india.png` | Same as F13 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase10_drawdowns_india.png`. |
| F19 | Vectorised research backtest | `reports/figures/phase_10/return_distribution_us.png` | Same as F13 | available-local | optional-selected | US research-proxy return distribution diagnostic. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F20 | Vectorised research backtest | `reports/figures/phase_10/return_distribution_india.png` | Same as F13 | available-local | optional-selected | India research-proxy return distribution diagnostic. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F21 | Cross-market analysis | `reports/figures/phase_13/us_india_vrp.png` | `python scripts/run_cross_market_analysis.py --model ALL --force` | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase13_us_india_vrp.png`. |
| F22 | Cross-market analysis | `reports/figures/phase_13/us_india_stress_prob.png` | Same as F21 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase13_us_india_stress_prob.png`. |
| F23 | Cross-market analysis | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Same as F21 | available-local | local-only | Original generated source figure; selected copy staged under `reports/final/figures/phase13_lagged_us_vs_india_stress.png`. |
| F24 | Cross-market analysis | `reports/figures/phase_13/india_overlay_equity_curves.png` | Same as F21 | available-local | optional-selected | Analysis-only India overlay research-proxy curve. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F25 | Cross-market analysis | `reports/figures/phase_13/india_overlay_exposure.png` | Same as F21 | available-local | optional-selected | Analysis-only India overlay exposure diagnostic. Optional/maybe; inspected in `local_artifacts/phase14_evidence_review.md`. |
| F26 | Final presentation / report | `reports/final/report_pipeline_diagram.png` | Optional manual/report helper | to-create | optional-selected | Optional only; Markdown text pipeline may be enough |
| F27 | Final PDF export | `reports/final/final_report.pdf` | PDF export from `reports/final/final_report.md` | to-create | commit | PDF is deliverable, not a figure |
| FF01 | Implied variance and VRP construction | `reports/final/figures/us_iv_rv_vrp.png` | Copied from `reports/figures/us_iv_rv_vrp.png` after Phase 14 evidence review | staged-final | commit | VIX proxy diagnostic; not variance swap quote |
| FF02 | Implied variance and VRP construction | `reports/final/figures/india_iv_rv_vrp.png` | Copied from `reports/figures/india_iv_rv_vrp.png` after Phase 14 evidence review | staged-final | commit | India VIX proxy diagnostic; not variance swap quote |
| FF03 | Vectorised research backtest | `reports/final/figures/phase10_equity_curves_common_start_us.png` | Copied from `reports/figures/phase_10/equity_curves_common_start_us.png` after Phase 14 evidence review | staged-final | commit | Research-proxy cumulative curve; not account equity |
| FF04 | Vectorised research backtest | `reports/final/figures/phase10_equity_curves_common_start_india.png` | Copied from `reports/figures/phase_10/equity_curves_common_start_india.png` after Phase 14 evidence review | staged-final | commit | Research-proxy cumulative curve; not account equity |
| FF05 | Vectorised research backtest | `reports/final/figures/phase10_drawdowns_us.png` | Copied from `reports/figures/phase_10/drawdowns_us.png` after Phase 14 evidence review | staged-final | commit | Research-proxy drawdown; not account drawdown |
| FF06 | Vectorised research backtest | `reports/final/figures/phase10_drawdowns_india.png` | Copied from `reports/figures/phase_10/drawdowns_india.png` after Phase 14 evidence review | staged-final | commit | Research-proxy drawdown; not account drawdown |
| FF07 | Cross-market analysis | `reports/final/figures/phase13_us_india_vrp.png` | Copied from `reports/figures/phase_13/us_india_vrp.png` after Phase 14 evidence review | staged-final | commit | Descriptive cross-market diagnostic; not causal proof |
| FF08 | Cross-market analysis | `reports/final/figures/phase13_us_india_stress_prob.png` | Copied from `reports/figures/phase_13/us_india_stress_prob.png` after Phase 14 evidence review | staged-final | commit | Descriptive cross-market diagnostic; not causal proof |
| FF09 | Cross-market analysis | `reports/final/figures/phase13_lagged_us_vs_india_stress.png` | Copied from `reports/figures/phase_13/lagged_us_vs_india_stress.png` after Phase 14 evidence review | staged-final | commit | Lagged predictive diagnostic; not causal proof |

## Figures Selected for Final Report Body

Final figure selection should be compact. Recommended maximum: 6-8 figures in the body, with additional figures referenced in appendix or inventory.

| Candidate | Source | Use |
|---|---|---|
| VRP construction figure | `reports/final/figures/us_iv_rv_vrp.png`; `reports/final/figures/india_iv_rv_vrp.png` | Show implied vs realised variance and VRP construction |
| Regime interpretation figure | Threshold/HMM/MAR selected figure or table-driven summary | Show regime separation without overclaiming |
| Phase 10 proxy curve | `reports/final/figures/phase10_equity_curves_common_start_us.png`; `reports/final/figures/phase10_equity_curves_common_start_india.png` | Show research-proxy comparison only |
| Drawdown figure | `reports/final/figures/phase10_drawdowns_us.png`; `reports/final/figures/phase10_drawdowns_india.png` | Show proxy drawdown behaviour |
| Cross-market VRP figure | `reports/final/figures/phase13_us_india_vrp.png` | Show dual-market comparison |
| Lagged diagnostic figure | `reports/final/figures/phase13_lagged_us_vs_india_stress.png` | Show predictive diagnostic, not causal proof |
| Overlay exposure figure | `reports/figures/phase_13/india_overlay_exposure.png` | Optional; analysis-only overlay |

## Figure Wording Constraints

Use:

```text
research-proxy cumulative curve
proxy drawdown
diagnostic figure
analysis-only overlay
lagged predictive diagnostic
```

Do not use:

```text
account equity curve
live trading curve
option PnL curve
causal transmission plot
execution result
paper trading result
```

## Figures Not to Include Directly

Do not include:

```text
unreadable huge plots
plots with broker-sensitive labels
exploratory notebook screenshots
figures not produced by documented scripts
figures not listed in this inventory
figures whose numeric claims are absent from result_claims_audit.md
```
