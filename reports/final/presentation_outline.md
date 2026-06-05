# Presentation Outline

Project:

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

This is an evidence-first presentation outline for the EPAT final submission. Each slide lists the intended message, supporting artifact path, and speaker note.

Do not insert numeric findings until the relevant local result tables have been inspected and the claim is recorded in:

```text
reports/final/result_claims_audit.md
```

---

## Slide 1 — Title and Objective

### Bullets

* Project title: **Variance Risk Premium Decomposition and Regime-Conditional Harvesting**
* Dual-market empirical study: US SPX/VIX and India NIFTY/India VIX.
* Objective: measure VRP, decompose it by volatility regime, and evaluate regime-conditioned research-proxy short-volatility exposure.
* Scope: empirical research pipeline, vectorised proxy backtest, cross-market diagnostics, and paper-signal readiness.
* Non-scope: live trading, true option-chain PnL, account returns, broker execution.

### Suggested figure/table

* None required.
* Optional: simple pipeline diagram from README/report.

### Source artifact path

```text
README.md
reports/final/final_report.md
reports/final/result_claims_audit.md
```

### Speaker note

This project is framed as a research and strategy-evaluation study, not as a deployed trading system. The main question is whether regime conditioning improves research-proxy VRP harvesting behaviour versus an unconditional benchmark.

---

## Slide 2 — Research Motivation

### Bullets

* Equity-index implied volatility often embeds compensation for protection demand, volatility uncertainty, and crash risk.
* Short-volatility exposure can harvest this premium but is vulnerable during stress regimes.
* Unconditional short-volatility harvesting can look attractive in calm periods while hiding tail risk.
* Regime conditioning is tested as a risk-control layer, not as a guarantee.
* Comparing the US and India adds cross-market empirical breadth.

### Suggested figure/table

* Optional conceptual diagram: implied variance minus realised variance.
* Optional VRP construction figure if selected.

### Source artifact path

```text
reports/final/final_report.md
reports/tables/vrp_summary.csv
reports/figures/us_iv_rv_vrp.png
reports/figures/india_iv_rv_vrp.png
```

### Speaker note

The motivation is not “short volatility always works.” The motivation is that VRP can exist structurally, but harvesting it without regime awareness can be exposed to severe drawdowns.

---

## Slide 3 — Data and Markets

### Bullets

* US market: SPX/SPY underlying, VIX implied-volatility proxy.
* India market: NIFTY 50 underlying, India VIX implied-volatility proxy.
* Frequency: daily public data.
* Realised variance is built from daily OHLC data.
* Generated raw and processed panels remain local and are not committed.

### Suggested figure/table

* Compact data coverage table.

### Source artifact path

```text
reports/tables/data_audit.csv
configs/data_sources.yaml
configs/markets.yaml
docs/reproducibility.md
```

### Speaker note

VIX and India VIX are used as implied-volatility proxies, not variance swap quotes. Daily OHLC realised variance is also a proxy, not true intraday realised variance.

---

## Slide 4 — VRP Construction

### Bullets

* Implied variance proxy: squared VIX or squared India VIX after percentage scaling.
* Realised variance proxy: daily OHLC estimators, with Garman-Klass 22-day annualised RV as the primary convention.
* VRP compares implied variance with realised variance or HAR-forecast realised variance.
* Forward ex-post VRP labels are evaluation outcomes, not tradable features.
* HAR-based prospective VRP is the tradable-time model-based feature.

### Suggested figure/table

* IV/RV/VRP time-series figure for US and India.
* Compact VRP summary table.

### Source artifact path

```text
reports/tables/vrp_summary.csv
reports/tables/vrp_metadata.json
reports/figures/us_iv_rv_vrp.png
reports/figures/india_iv_rv_vrp.png
reports/final/table_inventory.md
reports/final/figure_inventory.md
```

### Speaker note

The important distinction is between ex-post labels and signal-time features. Future realised variance is never used as a tradable input.

---

## Slide 5 — HAR-RV Forecast Layer

### Bullets

* HAR-RV forecasts provide a model-based realised variance estimate available at the forecast timestamp.
* The forecast layer supports prospective HAR-based VRP construction.
* Expanding or rolling training preserves point-in-time discipline.
* Forecast outputs are audited for no-lookahead constraints.
* Forecasts are model-dependent estimates, not guarantees.

### Suggested figure/table

* HAR forecast accuracy table.
* HAR no-lookahead audit summary.
* Optional HAR forecast figure.

### Source artifact path

```text
reports/tables/har_forecast_accuracy.csv
reports/tables/har_vrp_summary.csv
reports/tables/har_no_lookahead_audit.csv
reports/figures/har_forecast_us.png
reports/figures/har_forecast_india.png
```

### Speaker note

HAR-RV is included because VRP harvesting needs a prospective estimate of realised variance. Ex-post realised variance can evaluate outcomes but cannot be used to decide exposure.

---

## Slide 6 — Regime Model Ladder

### Bullets

* Threshold regimes: simple interpretable deterministic baseline.
* Gaussian HMM: latent regime classifier using filtered probabilities for time-safe use.
* Markov autoregression: AR-aware regime model for observed-series persistence.
* MSVOL: Python-only volatility-regime robustness appendix.
* Regime states are economic interpretations, not observed truth.

### Suggested figure/table

* Regime model ladder table.
* Compact state summary from Phase 5/6/7.
* Optional threshold/HMM/MAR diagnostic figure if selected.

### Source artifact path

```text
reports/tables/threshold_regime_summary.csv
reports/tables/phase_6/us/hmm_state_summary.csv
reports/tables/phase_6/india/hmm_state_summary.csv
reports/tables/phase_7/us/mar_state_summary.csv
reports/tables/phase_7/india/mar_state_summary.csv
reports/tables/phase_8/msvol_model_comparison_appendix.csv
```

### Speaker note

The regime ladder moves from interpretability to latent modelling to AR-aware modelling. Gaussian HMM does not directly model observed autocorrelation; MAR addresses that limitation in reduced-form form.

---

## Slide 7 — Strategy Signal Design

### Bullets

* Strategy layer converts regime and carry information into next-session exposure intentions.
* Fixed strategy universe is defined before Phase 10 evaluation.
* Strategy variants compare unconditional, threshold-conditioned, HMM-conditioned, MAR-conditioned, and carry-aware exposure rules.
* Signals are not broker orders.
* Signal timing is audited for no-lookahead constraints.

### Suggested figure/table

* Strategy signal summary table.
* Exposure-by-strategy table or compact signal design table.

### Source artifact path

```text
reports/tables/phase_9/strategy_signal_summary.csv
reports/tables/phase_9/strategy_no_lookahead_audit.csv
configs/strategies.yaml
```

### Speaker note

The strategy layer is intentionally separated from execution. It produces exposure intentions used by the research backtest, not order instructions.

---

## Slide 8 — Backtest Accounting

### Bullets

* Phase 10 is a vectorised research-proxy backtest.
* Returns are proxy units, not account returns.
* Cumulative curves are additive research-proxy curves, not executable equity curves.
* No option chain, contract sizing, margin, liquidity, or broker execution is modelled.
* Overlapping 22-day labels make annualised metrics approximate.

### Suggested figure/table

* Backtest accounting caveat table.
* Backtest availability summary.
* Backtest no-lookahead audit.

### Source artifact path

```text
reports/tables/phase_10/backtest_availability_summary.csv
reports/tables/phase_10/backtest_metadata.json
reports/tables/phase_10/backtest_no_lookahead_audit.csv
reports/tables/phase_10/phase10_final_audit.json
```

### Speaker note

This slide prevents the most dangerous overclaim. The backtest is useful for comparing research designs, but it is not a true options trading account simulation.

---

## Slide 9 — Main Results: Research-Proxy Backtest

### Bullets

* Compare unconditional benchmark against regime-conditioned variants.
* Discuss only inspected metrics from Phase 10 summary tables.
* Main result statement must remain in research-proxy wording.
* Drawdown and tail behaviour should be interpreted as tested-sample evidence.
* No live profitability or account-return claim.

### Suggested figure/table

* Compact `backtest_summary.csv` excerpt.
* Common-start research-proxy cumulative curve.
* Proxy drawdown figure.

### Source artifact path

```text
reports/tables/phase_10/backtest_summary.csv
reports/tables/phase_10/backtest_common_start_summary.csv
reports/tables/phase_10/backtest_tail_summary.csv
reports/figures/phase_10/equity_curves_common_start_us.png
reports/figures/phase_10/equity_curves_common_start_india.png
reports/figures/phase_10/drawdowns_us.png
reports/figures/phase_10/drawdowns_india.png
reports/final/result_claims_audit.md
```

### Speaker note

Use placeholders until the local tables are inspected. Correct wording: “In the research-proxy backtest, selected regime-conditioned variants show [verified relation] relative to the unconditional benchmark in the tested sample.”

---

## Slide 10 — Robustness and Stress Behaviour

### Bullets

* Robustness checks evaluate sensitivity to assumed proxy transaction costs.
* Subperiod tests check whether findings depend on a specific window.
* Crisis-window diagnostics describe historical stress-period proxy behaviour.
* Tail summaries help evaluate drawdown and left-tail exposure.
* Robustness is sensitivity evidence, not proof of future stability.

### Suggested figure/table

* Cost sensitivity table.
* Subperiod robustness table.
* Crisis-window performance table.
* Tail summary table.

### Source artifact path

```text
reports/tables/phase_10/robustness_cost_sensitivity.csv
reports/tables/phase_10/robustness_subperiods.csv
reports/tables/phase_10/crisis_window_performance.csv
reports/tables/phase_10/backtest_tail_summary.csv
```

### Speaker note

Do not say “robust to costs” unless the cost table supports that exact wording. Say “sensitivity to assumed costs” unless the evidence supports stronger language.

---

## Slide 11 — Cross-Market US-India Analysis

### Bullets

* Same-date diagnostics describe co-movement between US and Indian VRP/regime variables.
* Lagged-US diagnostics test predictive association with later Indian outcomes.
* Granger-style diagnostics are lead-lag diagnostics, not causal proof.
* Logistic tests evaluate incremental predictive information in the tested sample.
* India overlay is analysis-only and outside the locked Phase 9 strategy universe.

### Suggested figure/table

* US-India VRP figure.
* Stress probability comparison.
* Lead-lag table.
* Logistic model comparison table.
* India overlay summary only if clearly labelled as analysis-only.

### Source artifact path

```text
reports/tables/phase_13/alignment_audit.csv
reports/tables/phase_13/no_lookahead_audit.csv
reports/tables/phase_13/lead_lag_table.csv
reports/tables/phase_13/granger_diagnostics.csv
reports/tables/phase_13/logistic_model_comparison.csv
reports/tables/phase_13/india_overlay_summary.csv
reports/figures/phase_13/us_india_vrp.png
reports/figures/phase_13/us_india_stress_prob.png
reports/figures/phase_13/lagged_us_vs_india_stress.png
```

### Speaker note

This slide must be careful. Cross-market evidence can be predictive or descriptive, but it is not causal. The overlay is not a new implemented trading strategy.

---

## Slide 12 — Paper-Signal Readiness Appendix

### Bullets

* Phase 11 converts research signals into a guarded paper-signal format.
* It validates configuration, risk checks, and live-order blocking.
* It does not place broker orders.
* It does not report paper trading results.
* Phase 12 paper execution adapter is skipped / future optional.

### Suggested figure/table

* Live-order guard status.
* Risk-check summary.
* Integration validation summary.

### Source artifact path

```text
reports/tables/phase_11/risk_check_report.csv
reports/tables/phase_11/phase11_integration_report.json
reports/tables/phase_11/live_order_guard_report.json
reports/final/future_work.md
```

### Speaker note

Allowed wording: “paper-signal readiness layer,” “order-guard validation,” “configuration and risk-check demonstration,” and “no broker orders placed.” Do not use “paper trading results,” “execution layer,” “live strategy,” or “broker backtest.”

---

## Slide 13 — Limitations

### Bullets

* VIX and India VIX are implied-volatility proxies, not variance swap quotes.
* Daily OHLC RV estimators are proxies for true realised variance.
* Phase 10 returns are research-proxy units, not executable account returns.
* Phase 13 diagnostics are statistical/predictive diagnostics, not causal proof.
* Phase 11 did not place broker orders; Phase 12 was skipped / future optional.

### Suggested figure/table

* Terminology lock table.
* Final limitations summary table.

### Source artifact path

```text
reports/final/limitations.md
reports/final/final_report.md
reports/final/result_claims_audit.md
docs/known_limitations.md
```

### Speaker note

This is not a disclaimer slide to rush through. It defines what the project actually proves. It protects the credibility of the results.

---

## Slide 14 — Conclusion and Future Work

### Bullets

* The project delivers a complete dual-market public-data VRP research pipeline.
* It separates ex-post outcomes from tradable-time features.
* It evaluates a regime-conditioned short-volatility proxy framework under no-lookahead constraints.
* It adds cross-market diagnostics and paper-signal readiness without claiming execution.
* Future work: true option-chain PnL, intraday RV, true R MSGARCH, paper execution adapter, and production-grade broker safeguards.

### Suggested figure/table

* Final pipeline summary.
* Future work table.

### Source artifact path

```text
reports/final/future_work.md
reports/final/reproducibility_note.md
docs/submission_package.md
docs/release_checklist.md
```

### Speaker note

End with the correct scope: strong empirical research and release discipline, not live-trading deployment. The strongest contribution is the complete, no-lookahead, dual-market regime framework with explicit artifact governance.

---

# Appendix Slide A — Terminology Lock

Use only if additional slides are allowed.

| Term used                          | Meaning                                             | Not meant as                             |
| ---------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| Research-proxy return              | Additive VRP proxy backtest unit                    | Account return                           |
| Exposure intention                 | Signal target from strategy layer                   | Broker order                             |
| Filtered probability               | Time-t available regime probability                 | Full-sample smoothed probability         |
| Cross-market predictive diagnostic | Statistical lead-lag or predictive association test | Causal proof                             |
| MSVOL robustness                   | Python Markov-switching volatility check            | True R MSGARCH                           |
| Paper-signal readiness             | Signal-format and guard demonstration               | Broker execution or paper trading result |

Source:

```text
reports/final/final_report.md
reports/final/limitations.md
reports/final/result_claims_audit.md
```

---

# Appendix Slide B — Final Validation

Use only if the review panel asks about release readiness.

Validation commands:

```bash
git diff --check
git status --short
git ls-files | findstr /i "\.parquet \.pkl \.pickle \.joblib \.pt \.pth \.log \.env"
git ls-files | findstr /i "broker_cache data/raw data/interim data/processed"
pytest
python scripts/validate_phase11.py --print-json
python scripts/run_cross_market_analysis.py --validate-inputs-only
pytest tests/test_cross_market_alignment.py tests/test_cross_market_no_lookahead.py tests/test_cross_market_stats.py tests/test_cross_market_overlay.py tests/test_phase13_artifact_mutation.py tests/test_phase13_datetime_dtype.py
```

Source:

```text
docs/commands.md
docs/release_checklist.md
reports/final/reproducibility_note.md
```

Speaker note:

The validation package checks code health, artifact hygiene, broker-order guards, and cross-market no-lookahead assumptions.
