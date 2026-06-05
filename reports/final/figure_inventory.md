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
| F01 | Realised variance construction | `reports/figures/rv_estimators_us.png` | `python scripts/build_features.py --market ALL --feature rv --window 22` | placeholder | optional-selected | US RV estimator comparison |
| F02 | Realised variance construction | `reports/figures/rv_estimators_india.png` | Same as F01 | placeholder | optional-selected | India RV estimator comparison |
| F03 | Implied variance and VRP construction | `reports/figures/us_iv_rv_vrp.png` | `python scripts/build_features.py --market ALL --feature vrp` | placeholder | optional-selected | US IV/RV/VRP figure |
| F04 | Implied variance and VRP construction | `reports/figures/india_iv_rv_vrp.png` | Same as F03 | placeholder | optional-selected | India IV/RV/VRP figure |
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
| F15 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_common_start_us.png` | Same as F13 | available-local | optional-selected | Common-start US proxy curve |
| F16 | Vectorised research backtest | `reports/figures/phase_10/equity_curves_common_start_india.png` | Same as F13 | available-local | optional-selected | Common-start India proxy curve |
| F17 | Vectorised research backtest | `reports/figures/phase_10/drawdowns_us.png` | Same as F13 | available-local | optional-selected | US research-proxy drawdown diagnostic |
| F18 | Vectorised research backtest | `reports/figures/phase_10/drawdowns_india.png` | Same as F13 | available-local | optional-selected | India research-proxy drawdown diagnostic |
| F19 | Vectorised research backtest | `reports/figures/phase_10/return_distribution_us.png` | Same as F13 | available-local | optional-selected | US proxy return distribution |
| F20 | Vectorised research backtest | `reports/figures/phase_10/return_distribution_india.png` | Same as F13 | available-local | optional-selected | India proxy return distribution |
| F21 | Cross-market analysis | `reports/figures/phase_13/us_india_vrp.png` | `python scripts/run_cross_market_analysis.py --model ALL --force` | available-local | optional-selected | US-India VRP comparison |
| F22 | Cross-market analysis | `reports/figures/phase_13/us_india_stress_prob.png` | Same as F21 | available-local | optional-selected | Stress probability comparison |
| F23 | Cross-market analysis | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Same as F21 | available-local | optional-selected | Lagged-US diagnostic; not causal proof |
| F24 | Cross-market analysis | `reports/figures/phase_13/india_overlay_equity_curves.png` | Same as F21 | available-local | optional-selected | Analysis-only overlay proxy curve |
| F25 | Cross-market analysis | `reports/figures/phase_13/india_overlay_exposure.png` | Same as F21 | available-local | optional-selected | Analysis-only overlay exposure |
| F26 | Final presentation / report | `reports/final/report_pipeline_diagram.png` | Optional manual/report helper | to-create | optional-selected | Optional only; Markdown text pipeline may be enough |
| F27 | Final PDF export | `reports/final/final_report.pdf` | PDF export from `reports/final/final_report.md` | to-create | commit | PDF is deliverable, not a figure |

## Figures Selected for Final Report Body

Final figure selection should be compact. Recommended maximum: 6-8 figures in the body, with additional figures referenced in appendix or inventory.

| Candidate | Source | Use |
|---|---|---|
| VRP construction figure | `reports/figures/us_iv_rv_vrp.png`; `reports/figures/india_iv_rv_vrp.png` | Show implied vs realised variance and VRP construction |
| Regime interpretation figure | Threshold/HMM/MAR selected figure or table-driven summary | Show regime separation without overclaiming |
| Phase 10 proxy curve | `reports/figures/phase_10/equity_curves_common_start_us.png`; `reports/figures/phase_10/equity_curves_common_start_india.png` | Show research-proxy comparison only |
| Drawdown figure | `reports/figures/phase_10/drawdowns_us.png`; `reports/figures/phase_10/drawdowns_india.png` | Show proxy drawdown behaviour |
| Cross-market VRP figure | `reports/figures/phase_13/us_india_vrp.png` | Show dual-market comparison |
| Lagged diagnostic figure | `reports/figures/phase_13/lagged_us_vs_india_stress.png` | Show predictive diagnostic, not causal proof |
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
