# Result Claims Audit

This file maps final-report claims to evidence artifacts and prevents overclaiming.

No major conclusion should appear in `final_report.md`, `final_report.pdf`, `executive_summary.md`, or `presentation_outline.md` unless it is represented here.

## Audit Rules

1. Do not invent numbers.
2. Do not write numeric findings unless the evidence file has been inspected.
3. Use placeholders until evidence has been inspected.
4. Do not claim live-trading profitability.
5. Do not claim true option-chain PnL.
6. Do not claim account returns.
7. Do not claim causal US-to-India transmission from correlation, lead-lag, Granger-style, or logistic diagnostics.
8. Do not claim MSVOL is true MSGARCH.
9. Do not claim Phase 11 sent broker orders.
10. Do not imply Phase 12 was implemented.

## Placeholder Format

Use this format until a value is verified from a local result artifact:

```text
[INSERT VALUE FROM reports/tables/<path>: metric_name]
```

## Claims Table

| Claim | Evidence file | Evidence column / metric | Allowed wording | Forbidden overclaim | Report section |
|---|---|---|---|---|---|
| The project constructs a reproducible dual-market VRP research pipeline for the US and India. | `README.md`; `docs/phase_status.md`; `docs/reproducibility.md` | Phase status; pipeline docs | "The repository implements a reproducible public-data research pipeline for US and India VRP analysis." | "The repository implements a live trading system." | Executive summary; Methodology |
| VIX and India VIX are used as implied-volatility index proxies for implied variance. | `reports/tables/vrp_metadata.json`; `reports/tables/vrp_summary.csv` | implied-volatility source fields; IV columns | "VIX and India VIX are used as exchange-style implied-volatility proxies." | "The project uses true variance swap quotes." | Data; IV and VRP construction |
| Daily OHLC realised variance is a proxy for true realised variance. | `reports/tables/rv_summary.csv`; `reports/tables/rv_metadata.json` | RV estimator columns; primary estimator marker | "Realised variance is estimated from daily OHLC data, with Garman-Klass used as the primary proxy." | "The project observes true realised variance." | Realised variance construction |
| The 22-trading-day horizon is an approximate one-month research horizon. | `reports/tables/vrp_metadata.json`; `reports/tables/har_metadata.json`; `reports/tables/phase_10/backtest_metadata.json` | horizon; window; label horizon | "The project uses 22 trading days as an approximate one-month horizon." | "The labels are exactly 30-calendar-day variance swap outcomes." | Data; Methodology |
| Forward ex-post VRP labels are outcomes, not tradable features. | `reports/tables/vrp_metadata.json`; `reports/tables/phase_10/backtest_no_lookahead_audit.csv` | label construction; no-lookahead audit | "Forward ex-post VRP is used as an evaluation outcome." | "Forward realised VRP was used as a trading signal." | VRP construction; Backtest accounting |
| HAR-RV forecasts provide a model-based prospective realised variance estimate under point-in-time constraints. | `reports/tables/har_forecast_accuracy.csv`; `reports/tables/har_no_lookahead_audit.csv`; `reports/tables/har_vrp_summary.csv` | forecast error metrics; unavailable rows; audit result | "HAR-RV provides a model-dependent prospective RV forecast used to form HAR-based VRP features." | "HAR-RV predicts future variance with guaranteed accuracy." | HAR-RV forecasting |
| Threshold regimes provide an interpretable deterministic baseline. | `reports/tables/threshold_regime_summary.csv`; `reports/tables/threshold_vrp_by_state.csv`; `reports/tables/threshold_no_lookahead_audit.csv` | state counts; VRP by state; audit result | "Threshold regimes provide an interpretable baseline regime classification." | "Threshold regimes reveal the true market state." | Regime modelling |
| Gaussian HMM provides latent regime classification but does not directly model observed-series autocorrelation. | `reports/tables/phase_6/us/hmm_state_summary.csv`; `reports/tables/phase_6/india/hmm_state_summary.csv`; `reports/tables/phase_6/us/hmm_no_lookahead_audit.csv`; `reports/tables/phase_6/india/hmm_no_lookahead_audit.csv` | state summary; filtered probability audit; no-lookahead audit | "Gaussian HMM gives latent volatility/VRP regime structure using filtered probabilities for time-safe use." | "Gaussian HMM directly models autocorrelation in observed VRP." | Regime modelling |
| Markov autoregression improves the regime ladder by allowing observed-series autoregressive dynamics. | `reports/tables/phase_7/us/mar_state_summary.csv`; `reports/tables/phase_7/india/mar_state_summary.csv`; `reports/tables/phase_7/us/mar_ar_stability.csv`; `reports/tables/phase_7/india/mar_ar_stability.csv`; `reports/tables/phase_7/us/mar_no_lookahead_audit.csv`; `reports/tables/phase_7/india/mar_no_lookahead_audit.csv` | AR coefficients; state summaries; no-lookahead audit | "MAR extends the regime stack by modelling state-dependent autoregressive behaviour." | "MAR identifies true causal regimes." | Regime modelling |
| MSVOL is used only as Python-only volatility-regime robustness, not true R MSGARCH. | `reports/tables/phase_8/msvol_model_comparison_appendix.csv`; `reports/tables/phase_8/msvol_no_lookahead_audit.csv` | comparison fields; audit result | "MSVOL is a Python-only Markov-switching volatility robustness appendix." | "The project implements true MSGARCH." | MSVOL appendix |
| Strategy outputs are exposure intentions, not broker orders. | `reports/tables/phase_9/strategy_signal_summary.csv`; `reports/tables/phase_9/strategy_no_lookahead_audit.csv`; `reports/tables/phase_11/live_order_guard_report.json` | exposure columns; signal audit; live order guard | "Strategy rules produce next-session exposure intentions." | "The strategy placed executable broker orders." | Strategy construction; Phase 11 appendix |
| Phase 10 evaluates research-layer VRP proxy performance, not executable option-trading account returns. | `reports/tables/phase_10/backtest_summary.csv`; `reports/tables/phase_10/backtest_metadata.json`; `reports/tables/phase_10/phase10_final_audit.json` | strategy metrics; metadata; final audit | "Phase 10 reports vectorised research-proxy backtest results." | "Phase 10 reports realised account returns or option-chain PnL." | Vectorised research backtest |
| MAR probability sizing can be discussed only if supported by Phase 10 metrics. | `reports/tables/phase_10/backtest_summary.csv`; `reports/tables/phase_10/backtest_common_start_summary.csv`; `reports/tables/phase_10/backtest_tail_summary.csv` | strategy name; Sharpe/Sortino/drawdown/proxy return metrics | "In the research-proxy backtest, MAR probability sizing showed [INSERT VALUE/RELATION FROM reports/tables/phase_10/backtest_summary.csv]." | "MAR is a profitable live trading strategy." | Main findings |
| Cost sensitivity must be described as robustness over research-proxy assumptions. | `reports/tables/phase_10/robustness_cost_sensitivity.csv`; `reports/tables/phase_10/robustness_metadata.json` | cost-bps scenarios; strategy metrics | "The robustness check evaluates sensitivity to assumed transaction-cost levels in proxy units." | "The strategy is robust to all real execution costs." | Robustness checks |
| Crisis-window and tail diagnostics describe historical research-proxy behaviour only. | `reports/tables/phase_10/crisis_window_performance.csv`; `reports/tables/phase_10/backtest_tail_summary.csv` | crisis windows; drawdown/tail metrics | "Crisis-window diagnostics show how proxy returns behaved during selected stress windows." | "The strategy is protected against future crises." | Backtest; Robustness |
| Phase 11 demonstrates paper-signal readiness and live-order guards only. | `reports/tables/phase_11/risk_check_report.csv`; `reports/tables/phase_11/phase11_integration_report.json`; `reports/tables/phase_11/live_order_guard_report.json` | guard status; risk status; `live_order_sent` | "Phase 11 validates paper-signal readiness and live-order guard behaviour." | "Phase 11 ran paper trading results or broker execution." | IBKR paper-signal readiness appendix |
| Phase 12 was skipped and remains future optional work. | `docs/phase_status.md`; `reports/final/future_work.md` | phase status | "Phase 12 was intentionally skipped and left as future optional work." | "Phase 12 was implemented." | Future work; Appendix |
| Same-date US-India diagnostics are descriptive only. | `reports/tables/phase_13/vrp_level_correlations.csv`; `reports/tables/phase_13/vrp_change_correlations.csv`; `reports/tables/phase_13/regime_probability_correlations.csv`; `reports/tables/phase_13/state_label_agreement.csv` | correlations; agreement metrics | "Same-date cross-market diagnostics describe contemporaneous co-movement." | "Same-date correlation proves causal transmission." | Cross-market analysis |
| Lagged-US cross-market diagnostics are predictive/statistical diagnostics, not causal proof. | `reports/tables/phase_13/lead_lag_table.csv`; `reports/tables/phase_13/granger_diagnostics.csv`; `reports/tables/phase_13/logistic_model_comparison.csv`; `reports/tables/phase_13/logistic_oos_diagnostics.csv` | lagged metrics; Granger-style diagnostics; logistic OOS metrics | "Lagged-US diagnostics test whether prior US information has predictive association with India outcomes in the sample." | "US volatility causes Indian VRP regime transitions." | Cross-market analysis |
| Phase 13 India overlay is analysis-only and not part of the locked Phase 9 strategy universe. | `reports/tables/phase_13/india_overlay_summary.csv`; `reports/figures/phase_13/india_overlay_equity_curves.png`; `reports/figures/phase_13/india_overlay_exposure.png` | overlay metrics; exposure diagnostics | "The India overlay is an analysis-only diagnostic outside the locked Phase 9 strategy universe." | "The overlay is a new implemented production strategy." | Cross-market analysis |
| Generated panels remain local and are not committed to GitHub. | `docs/generated_artifact_policy.md`; `docs/artifact_inventory.md`; `.gitignore` | commit policy; local-only rules | "Generated panels remain local, with summaries/inventories used for review." | "All research data is included in the repository." | Reproducibility |
| Final report PDF is an export from the Markdown source after claims audit. | `reports/final/README.md`; `docs/final_report_checklist.md`; `docs/release_checklist.md` | PDF checklist; claims audit status | "The PDF is exported from the audited Markdown report source." | "The PDF is an independent report source with unaudited changes." | Reproducibility; Appendix |

## Claims Requiring Local Table Inspection

Fill this section only after inspecting local artifacts.

| Claim area | Required file | Required inspection |
|---|---|---|
| Main Phase 10 strategy ranking | `reports/tables/phase_10/backtest_summary.csv` | Identify strategies, risk-adjusted metrics, drawdown/tail metrics, and common sample caveats |
| Cost robustness | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | Confirm cost-bps grid and direction of metric sensitivity |
| Subperiod robustness | `reports/tables/phase_10/robustness_subperiods.csv` | Confirm sample segments and strategy stability |
| Phase 11 live-order guard | `reports/tables/phase_11/live_order_guard_report.json` | Confirm no live orders and guard status |
| Phase 13 predictive diagnostics | `reports/tables/phase_13/logistic_model_comparison.csv` | Confirm whether lagged-US features add predictive value in the tested sample |
| Phase 13 no-lookahead | `reports/tables/phase_13/no_lookahead_audit.csv` | Confirm strict lagged-US alignment |
| Phase 13 overlay | `reports/tables/phase_13/india_overlay_summary.csv` | Confirm analysis-only overlay metrics and wording |

## Forbidden Report Phrases

Do not use these phrases in final report, PDF, executive summary, or slides:

```text
profitable live strategy
account return
option-chain PnL
real trading returns
broker execution results
paper trading results
causal transmission
US causes India
true MSGARCH
Phase 12 implementation
orders were sent
guaranteed risk reduction
```

## Approved Replacement Phrases

Use these instead:

```text
research-proxy backtest
proxy return units
exposure intention
paper-signal readiness
live-order guard
predictive/statistical diagnostic
lead-lag association
Python-only MSVOL robustness
future optional paper execution adapter
no broker orders placed
drawdown control in the tested proxy sample
```
