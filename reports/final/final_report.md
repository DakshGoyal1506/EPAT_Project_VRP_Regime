# Variance Risk Premium Decomposition and Regime-Conditional Harvesting

## A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India

**EPAT Final Project Report**  
**Author:** Daksh Goyal  
**Programme:** Executive Programme in Algorithmic Trading  
**Project type:** Empirical volatility research, regime modelling, and research-layer strategy evaluation  
**Report source:** `reports/final/final_report.md`  
**PDF export:** `reports/final/final_report.pdf`

---

## Report Control Statement

This report is the final Markdown source for the EPAT submission package.

The project is a public-data empirical research project. It measures and studies the variance risk premium across the US and Indian equity-index volatility markets, evaluates regime-conditioned research-proxy harvesting rules, and documents a paper-signal readiness layer.

This report does not claim live-trading profitability, true option-chain PnL, account returns, broker execution, or causal cross-market transmission. Generated data panels remain local and are not committed to GitHub.

Every major result claim in this report must be traceable to:

```text
reports/final/result_claims_audit.md
```

Every table and figure used in this report must be listed in:

```text
reports/final/table_inventory.md
reports/final/figure_inventory.md
reports/final/selected_artifacts.md
```

---

## Abstract

The variance risk premium is the empirical spread between implied variance and subsequently realised variance. In equity-index markets, implied volatility indices such as VIX and India VIX often stand above subsequent realised volatility, but unconditional short-volatility exposure can suffer severe losses during stress periods.

This project builds a reproducible dual-market research pipeline for the US and India. The US leg uses SPX/SPY and VIX. The India leg uses NIFTY 50 and India VIX. The pipeline constructs realised variance from daily OHLC data, converts implied-volatility indices into annualised implied variance proxies, builds variance risk premium features and forward outcome labels, estimates HAR-RV forecasts, fits threshold, Gaussian HMM, Markov autoregression, and MSVOL-style robustness regime models, constructs regime-conditioned exposure intentions, evaluates a vectorised research-proxy backtest, and performs cross-market US-India diagnostics.

The empirical objective is not to prove live tradability. It is to test whether regime-conditioned exposure improves research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting, and whether US and Indian VRP regimes display useful descriptive or predictive relationships. Phase 11 adds a paper-signal readiness appendix with live-order guards, but no broker orders are placed. Phase 12 paper execution is intentionally skipped and left as future optional work.

---

## Executive Summary

This project studies whether variance risk premium harvesting can be made more defensible by conditioning exposure on volatility regimes.

The research design has three central components:

1. **VRP measurement:** estimate realised variance from daily OHLC data and compare it with implied variance proxies derived from VIX and India VIX.
2. **Regime decomposition:** classify market states using threshold regimes, Gaussian HMM, Markov autoregression, and MSVOL robustness diagnostics.
3. **Research-proxy strategy evaluation:** compare unconditional short-volatility harvesting with regime-conditioned exposure intentions in a vectorised backtest.

The project is dual-market by design. It evaluates the US and Indian markets separately, then adds a cross-market analysis layer to study same-date co-movement, lagged-US predictive diagnostics, and an analysis-only India overlay.

Main evidence categories:

| Evidence area                | Primary source                                                                                                 |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Data and VRP construction    | `reports/tables/vrp_summary.csv`; `reports/tables/vrp_metadata.json`                                           |
| HAR forecast layer           | `reports/tables/har_forecast_accuracy.csv`; `reports/tables/har_no_lookahead_audit.csv`                        |
| Regime model ladder          | Phase 5, 6, 7, and 8 diagnostic tables                                                                         |
| Strategy signal construction | `reports/tables/phase_9/strategy_signal_summary.csv`                                                           |
| Research-proxy backtest      | `reports/tables/phase_10/backtest_summary.csv`                                                                 |
| Robustness checks            | `reports/tables/phase_10/robustness_cost_sensitivity.csv`; `reports/tables/phase_10/robustness_subperiods.csv` |
| Paper-signal readiness       | `reports/tables/phase_11/live_order_guard_report.json`                                                         |
| Cross-market analysis        | `reports/tables/phase_13/logistic_model_comparison.csv`; `reports/tables/phase_13/lead_lag_table.csv`          |

Numeric findings are intentionally left as placeholders until the final local artifacts are inspected.

---

## 1. Motivation

Variance risk premium research sits between market microstructure, volatility forecasting, and systematic risk-premia design.

Equity-index options often embed a premium for protection demand, crash risk, jump risk, and volatility uncertainty. Short-volatility strategies attempt to harvest that premium, but unconditional exposure is structurally vulnerable to volatility spikes. The central motivation of this project is to evaluate whether regime conditioning can retain useful premium exposure while reducing exposure during historically adverse states.

The project is designed for EPAT as both a research and implementation exercise:

* It uses public daily data rather than private option-chain data.
* It separates research-proxy evaluation from executable trading.
* It enforces no-lookahead constraints.
* It compares two markets with different structures.
* It adds a paper-signal readiness appendix without crossing into broker execution.

---

## 2. Research Questions

The project is built around the following questions.

### RQ1 — VRP existence and construction

Can VIX and India VIX be used as implied-volatility proxies to construct a consistent public-data VRP panel for the US and Indian markets?

### RQ2 — Forecasting layer

Does a HAR-RV model provide a useful prospective realised-variance forecast layer for constructing HAR-based VRP features under point-in-time constraints?

### RQ3 — Regime decomposition

Do threshold, Gaussian HMM, and Markov autoregression models produce economically interpretable volatility or VRP regimes?

### RQ4 — Regime-conditioned harvesting

Do regime-conditioned exposure rules improve research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting in the tested sample?

### RQ5 — Robustness

Are results sensitive to assumed transaction costs, subperiods, crisis windows, model choice, and sample alignment?

### RQ6 — Cross-market structure

Do US and Indian VRP/regime diagnostics display descriptive co-movement or lagged predictive associations, without implying causal transmission?

### RQ7 — Operational readiness

Can the final research signal be converted into a guarded paper-signal format without placing broker orders?

---

## 3. Data

### 3.1 Markets

| Market | Underlying | Implied-volatility proxy |
| ------ | ---------- | ------------------------ |
| US     | SPX / SPY  | VIX                      |
| India  | NIFTY 50   | India VIX                |

### 3.2 Data sources

The project uses public daily market data. Data loaders support official and fallback sources where possible.

Primary data categories:

| Data category            | US                        | India                     |
| ------------------------ | ------------------------- | ------------------------- |
| Underlying OHLC          | SPX/SPY daily OHLC        | NIFTY 50 daily OHLC       |
| Implied-volatility proxy | VIX                       | India VIX                 |
| Frequency                | Daily                     | Daily                     |
| Storage policy           | Local generated artifacts | Local generated artifacts |

### 3.3 Data caveats

The data layer has important limitations:

1. VIX and India VIX are implied-volatility index proxies, not variance swap quotes.
2. Daily OHLC data is not full intraday realised variance.
3. Yahoo Finance data is convenient but not official exchange data.
4. NSE scripted access may be blocked; manual CSV override is supported.
5. Generated raw and processed data panels remain local and are not committed.

### 3.4 Data coverage

Insert final inspected data coverage table here.

```text
[INSERT COMPACT TABLE FROM reports/tables/data_audit.csv]
```

---

## 4. Methodology Overview

The project follows this pipeline:

```text
public daily data
  ↓
clean OHLC / VIX / India VIX series
  ↓
realised variance estimators
  ↓
implied variance construction
  ↓
variance risk premium construction
  ↓
HAR-RV forecast
  ↓
threshold regimes
  ↓
Gaussian HMM regimes
  ↓
Markov autoregression regimes
  ↓
MSVOL robustness appendix
  ↓
regime-conditioned strategy signals
  ↓
vectorised research backtest
  ↓
robustness checks
  ↓
IBKR paper-signal readiness appendix
  ↓
cross-market US-India analysis
  ↓
final report and release package
```

The design separates four layers:

| Layer                  | Purpose                                  | Output type                                |
| ---------------------- | ---------------------------------------- | ------------------------------------------ |
| Data and feature layer | Build RV, IV, VRP, HAR-VRP features      | Local panels and summary tables            |
| Regime layer           | Estimate market states                   | Regime probabilities and diagnostics       |
| Strategy layer         | Convert regimes into exposure intentions | Signal panels                              |
| Evaluation layer       | Evaluate research-proxy behaviour        | Backtest diagnostics and robustness tables |

---

## 5. Realised Variance Construction

Realised variance is estimated from daily OHLC data. The project includes multiple estimators, with Garman-Klass 22-trading-day annualised realised variance used as the primary proxy.

Candidate estimators include:

| Estimator       | Inputs                      | Role                         |
| --------------- | --------------------------- | ---------------------------- |
| Close-to-close  | Close prices                | Baseline                     |
| Parkinson       | High/low range              | Range-based robustness       |
| Garman-Klass    | Open/high/low/close         | Primary daily OHLC estimator |
| Rogers-Satchell | Open/high/low/close         | Drift-robust range estimator |
| Yang-Zhang      | Gap and intraday components | Optional robustness          |

The primary realised variance convention is:

```text
rv_gk_22d_ann
```

This should be interpreted as a daily OHLC-based proxy for annualised realised variance over a 22-trading-day window. It is not a directly observed realised variance swap outcome.

Insert final RV summary here:

```text
[INSERT VALUE/TABLE FROM reports/tables/rv_summary.csv]
```

Selected figure candidate:

```text
[INSERT FIGURE IF SELECTED: reports/figures/rv_estimators_us.png]
[INSERT FIGURE IF SELECTED: reports/figures/rv_estimators_india.png]
```

---

## 6. Implied Variance and VRP Construction

VIX and India VIX are converted into implied variance proxies by squaring the volatility index after percentage scaling.

Conceptually:

```text
implied_variance_proxy = (implied_volatility_index / 100)^2
```

The project compares implied variance proxies with realised variance proxies to construct VRP features.

The project uses two VRP concepts:

| Concept                   | Meaning                                                              | Tradable at signal time?     |
| ------------------------- | -------------------------------------------------------------------- | ---------------------------- |
| Ex-post forward VRP       | Implied variance at time t minus future realised variance outcome    | No                           |
| HAR-based prospective VRP | Implied variance at time t minus HAR-RV forecast available at time t | Yes as a model-based feature |

Forward ex-post labels are outcome labels for evaluation. They are not used as tradable features.

Insert final VRP construction summary here:

```text
[INSERT VALUE/TABLE FROM reports/tables/vrp_summary.csv]
[INSERT METADATA FROM reports/tables/vrp_metadata.json]
```

Selected figure candidates:

```text
[INSERT FIGURE IF SELECTED: reports/figures/us_iv_rv_vrp.png]
[INSERT FIGURE IF SELECTED: reports/figures/india_iv_rv_vrp.png]
```

---

## 7. HAR-RV Forecasting

The HAR-RV layer estimates prospective realised variance using lagged realised-variance features. Its role is to provide a model-based expected realised variance input for prospective VRP construction.

The HAR forecast layer is constrained by point-in-time rules:

1. Forecasts are estimated with expanding or rolling training windows.
2. Future realised variance is not used as a feature.
3. Forecast availability is audited.
4. HAR outputs are model-dependent estimates, not guarantees.

Insert final HAR summary here:

```text
[INSERT VALUE/TABLE FROM reports/tables/har_forecast_accuracy.csv]
[INSERT VALUE/TABLE FROM reports/tables/har_vrp_summary.csv]
[INSERT AUDIT STATUS FROM reports/tables/har_no_lookahead_audit.csv]
```

---

## 8. Regime Modelling

The regime modelling ladder is intentionally progressive. Each layer addresses a different modelling need.

| Model layer           | Purpose                       | Limitation                                              |
| --------------------- | ----------------------------- | ------------------------------------------------------- |
| Threshold regimes     | Simple interpretable baseline | Deterministic and threshold-sensitive                   |
| Gaussian HMM          | Latent state classification   | Does not directly model observed-series autocorrelation |
| Markov autoregression | AR-aware regime model         | Reduced-form and numerically sensitive                  |
| MSVOL appendix        | Volatility-regime robustness  | Python-only MSVOL, not true R MSGARCH                   |

### 8.1 Threshold regimes

Threshold regimes provide a deterministic benchmark. They are useful because a complex regime model should be compared against a simple interpretable baseline.

Insert final threshold-regime evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/threshold_regime_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/threshold_vrp_by_state.csv]
[INSERT AUDIT STATUS FROM reports/tables/threshold_no_lookahead_audit.csv]
```

### 8.2 Gaussian HMM

The Gaussian HMM estimates latent regimes from observed features. Filtered probabilities are used where time-safe probabilities are required. Full-sample smoothed probabilities are diagnostic only.

The HMM layer should be described as a latent regime classifier, not as a true market-state oracle.

Insert final HMM evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_6/us/hmm_state_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_6/india/hmm_state_summary.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_6/us/hmm_no_lookahead_audit.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_6/india/hmm_no_lookahead_audit.csv]
```

### 8.3 Markov autoregression

Markov autoregression extends the regime ladder by allowing state-dependent autoregressive dynamics in the observed series. This addresses an important limitation of standard Gaussian HMM emissions.

MAR remains a reduced-form model. Its regimes are economic interpretations, not observed ground truth.

Insert final MAR evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_7/us/mar_state_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_7/india/mar_state_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_7/us/mar_ar_stability.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_7/india/mar_ar_stability.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_7/us/mar_no_lookahead_audit.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_7/india/mar_no_lookahead_audit.csv]
```

### 8.4 MSVOL robustness appendix

The MSVOL layer is a Python-only Markov-switching volatility robustness appendix. It is not a true R MSGARCH implementation and is not used for strategy construction or backtesting.

Insert final MSVOL evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_8/msvol_model_comparison_appendix.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_8/msvol_no_lookahead_audit.csv]
```

---

## 9. Strategy Construction

The strategy layer converts regime and carry information into next-session exposure intentions.

The strategy outputs are not broker orders. They are research-layer exposure targets used for vectorised backtesting.

Core strategy categories:

| Strategy type           | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| Unconditional benchmark | Always maintains short-volatility proxy exposure                 |
| Threshold-conditioned   | Uses threshold regimes to reduce/suppress exposure               |
| HMM-conditioned         | Uses filtered HMM probabilities                                  |
| MAR-conditioned         | Uses filtered MAR probabilities                                  |
| Carry-aware variants    | Combine regime state and prospective HAR-based VRP/carry filters |

Insert final signal evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_9/strategy_signal_summary.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_9/strategy_no_lookahead_audit.csv]
```

---

## 10. Vectorised Research Backtest

The Phase 10 backtest is a vectorised research-proxy evaluation. It is not an executable option-chain simulation and does not report account returns.

The backtest evaluates proxy return units based on the project's VRP outcome construction and exposure intentions.

Important accounting caveats:

1. Returns are research-layer proxy units.
2. Cumulative curves are additive proxy curves, not account equity curves.
3. No initial capital, margin, option contract sizing, or broker execution is modelled.
4. Transaction costs are assumptions in proxy units.
5. Overlapping 22-day labels make annualised metrics approximate.

Insert final Phase 10 summary here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_10/backtest_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_10/backtest_common_start_summary.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_10/backtest_tail_summary.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_10/backtest_no_lookahead_audit.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_10/phase10_final_audit.json]
```

Selected figure candidates:

```text
[INSERT FIGURE IF SELECTED: reports/figures/phase_10/equity_curves_common_start_us.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_10/equity_curves_common_start_india.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_10/drawdowns_us.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_10/drawdowns_india.png]
```

All captions must use "research-proxy" language.

---

## 11. Robustness Checks

The robustness layer tests whether Phase 10 findings are sensitive to assumptions and sample choices.

Robustness categories:

| Robustness area          | Evidence                                                  |
| ------------------------ | --------------------------------------------------------- |
| Cost sensitivity         | `reports/tables/phase_10/robustness_cost_sensitivity.csv` |
| Subperiod behaviour      | `reports/tables/phase_10/robustness_subperiods.csv`       |
| Crisis windows           | `reports/tables/phase_10/crisis_window_performance.csv`   |
| Tail behaviour           | `reports/tables/phase_10/backtest_tail_summary.csv`       |
| Tradable proxy detection | `reports/tables/phase_10/tradable_proxy_detection.json`   |

Insert final robustness summary here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_10/robustness_cost_sensitivity.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_10/robustness_subperiods.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_10/crisis_window_performance.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_10/tradable_proxy_detection.json]
```

Robustness findings must not be phrased as proof of future trading performance.

---

## 12. Cross-Market US-India Analysis

Phase 13 adds a cross-market analysis layer. It is analysis-only and does not alter the locked Phase 9 strategy universe.

The cross-market layer includes:

| Analysis type                     | Interpretation                                         |
| --------------------------------- | ------------------------------------------------------ |
| Same-date diagnostics             | Descriptive co-movement only                           |
| Lagged-US diagnostics             | Predictive/statistical association only                |
| Granger-style diagnostics         | Lead-lag diagnostics, not causal proof                 |
| Logistic incremental-signal tests | Statistical predictive diagnostics                     |
| India overlay                     | Analysis-only overlay outside locked strategy universe |

Insert final cross-market evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_13/alignment_audit.csv]
[INSERT AUDIT STATUS FROM reports/tables/phase_13/no_lookahead_audit.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/vrp_level_correlations.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/lead_lag_table.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/granger_diagnostics.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/logistic_model_comparison.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/logistic_oos_diagnostics.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_13/india_overlay_summary.csv]
```

Selected figure candidates:

```text
[INSERT FIGURE IF SELECTED: reports/figures/phase_13/us_india_vrp.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_13/us_india_stress_prob.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_13/lagged_us_vs_india_stress.png]
[INSERT FIGURE IF SELECTED: reports/figures/phase_13/india_overlay_exposure.png]
```

The correct wording is:

```text
predictive/statistical diagnostic
lead-lag association
analysis-only overlay
```

The forbidden wording is:

```text
causal transmission
US causes India
new strategy implementation
live-trading evidence
```

---

## 13. IBKR Paper-Signal Readiness Appendix

Phase 11 is an operational readiness appendix. It converts research signals into guarded paper-signal artifacts and validates that live-order pathways remain blocked.

Allowed wording:

```text
paper-signal readiness layer
order-guard validation
configuration and risk-check demonstration
no broker orders placed
```

Forbidden wording:

```text
paper trading results
execution layer
live strategy
broker backtest
```

Insert final Phase 11 evidence here:

```text
[INSERT VALUE/TABLE FROM reports/tables/phase_11/risk_check_report.csv]
[INSERT VALUE/TABLE FROM reports/tables/phase_11/phase11_integration_report.json]
[INSERT VALUE/TABLE FROM reports/tables/phase_11/live_order_guard_report.json]
```

No broker-sensitive fields should be included in this report.

---

## 14. Main Findings

This section should be completed only after local evidence inspection.

### Finding 1 — VRP construction validity

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/vrp_summary.csv AND reports/tables/vrp_metadata.json]
```

Allowed form:

```text
The project successfully constructs aligned US and India VRP panels using VIX/India VIX implied-variance proxies and daily OHLC realised-variance proxies.
```

### Finding 2 — HAR forecast usefulness

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/har_forecast_accuracy.csv]
```

Allowed form:

```text
The HAR-RV layer provides a point-in-time model-based realised-variance forecast that supports prospective VRP construction.
```

### Finding 3 — Regime interpretability

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING PHASE 5/6/7 STATE SUMMARY TABLES]
```

Allowed form:

```text
The regime ladder produces economically interpretable calm/stress classifications, with MAR adding an AR-aware layer beyond the Gaussian HMM.
```

### Finding 4 — Research-proxy strategy performance

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/phase_10/backtest_summary.csv]
```

Allowed form:

```text
In the research-proxy backtest, selected regime-conditioned variants show [INSERT VERIFIED RELATION] relative to the unconditional benchmark in the tested sample.
```

Forbidden form:

```text
The strategy is profitable in live trading.
```

### Finding 5 — Robustness and cost sensitivity

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/phase_10/robustness_cost_sensitivity.csv AND robustness_subperiods.csv]
```

Allowed form:

```text
Robustness diagnostics show how the research-proxy results vary across cost assumptions and subperiods.
```

### Finding 6 — Cross-market evidence

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/phase_13/logistic_model_comparison.csv AND lead_lag_table.csv]
```

Allowed form:

```text
Cross-market diagnostics indicate [INSERT VERIFIED ASSOCIATION], interpreted as statistical/predictive evidence rather than causal proof.
```

### Finding 7 — Paper-signal readiness

Placeholder:

```text
[INSERT CLAIM AFTER INSPECTING reports/tables/phase_11/live_order_guard_report.json]
```

Allowed form:

```text
The Phase 11 appendix validates paper-signal readiness and live-order guard behaviour; no broker orders were placed.
```

---

## 15. Terminology Lock

The report uses the following terms in a restricted way.

| Term used in report                | Meaning                                             | Not meant as                             |
| ---------------------------------- | --------------------------------------------------- | ---------------------------------------- |
| Research-proxy return              | Additive VRP proxy backtest unit                    | Account return                           |
| Exposure intention                 | Signal target from the strategy layer               | Broker order                             |
| Filtered probability               | Time-t available regime probability                 | Full-sample smoothed probability         |
| Cross-market predictive diagnostic | Statistical lead-lag or predictive association test | Causal transmission proof                |
| MSVOL robustness                   | Python Markov-switching volatility check            | True R MSGARCH                           |
| Paper-signal readiness             | Guarded signal-format and risk-check demonstration  | Broker execution or paper trading result |

---

## 16. Limitations

The full final limitations document is:

```text
reports/final/limitations.md
```

Core limitations:

1. VIX and India VIX are implied-volatility proxies, not variance swap quotes.
2. Daily OHLC realised variance estimators are proxies for true realised variance.
3. A 22-trading-day horizon approximates a one-month horizon but is not identical to 30 calendar days.
4. Forward ex-post VRP labels are outcomes only, not tradable features.
5. HAR forecasts are model-dependent and constrained by training-window choices.
6. Gaussian HMM does not directly model observed-series autocorrelation.
7. Markov autoregression addresses observed-series autocorrelation but remains reduced-form.
8. MSVOL is Python-only volatility-regime robustness, not true R MSGARCH.
9. Strategy outputs are exposure intentions, not broker orders.
10. Phase 10 returns are research-layer proxy units, not account returns.
11. Overlapping 22-day outcome labels make annualised metrics approximate.
12. Phase 11 did not place broker orders.
13. Phase 12 paper execution was skipped and remains future optional work.
14. Phase 13 diagnostics are statistical/predictive diagnostics, not causal proof.
15. Generated panels remain local and are not committed to GitHub.

---

## 17. Future Work

The full future work document is:

```text
reports/final/future_work.md
```

Future extensions:

1. True option-chain PnL using historical option chains.
2. Instrument-level execution modelling with bid/ask spreads, liquidity, margin, and contract selection.
3. Optional IBKR paper execution adapter.
4. True R MSGARCH implementation.
5. Intraday realised variance using high-frequency data.
6. Broader international volatility-index comparison.
7. Production-grade monitoring and deployment safeguards.

These are explicitly outside the current EPAT submission scope.

---

## 18. Reproducibility

The full reproducibility documentation is:

```text
docs/reproducibility.md
reports/final/reproducibility_note.md
docs/commands.md
```

A reviewer can inspect the tracked repository without receiving large local data files. The reproducible tracked layer includes:

```text
source code
configs
scripts
tests
docs
README files
environment template
directory placeholders
final report package
```

Local-only generated artifacts include:

```text
raw market data
processed feature panels
trained model outputs
regime panels
strategy signal panels
backtest panels
broker/paper-signal cache
cross-market panels
```

Minimal validation commands:

```bash
pip install -e .
pytest
python scripts/download_data.py --dry-run
```

Final release validation commands are documented in:

```text
docs/release_checklist.md
docs/commands.md
```

---

## 19. Appendix A — Report Artifact Map

| Report area        | Evidence inventory                     |
| ------------------ | -------------------------------------- |
| Claims audit       | `reports/final/result_claims_audit.md` |
| Table inventory    | `reports/final/table_inventory.md`     |
| Figure inventory   | `reports/final/figure_inventory.md`    |
| Selected artifacts | `reports/final/selected_artifacts.md`  |
| Release checklist  | `docs/release_checklist.md`            |
| Submission map     | `docs/submission_package.md`           |

---

## 20. Appendix B — Required Final Validation

Before freezing Phase 14, run:

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

Expected policy result:

```text
No heavy generated artifacts tracked.
No credentials tracked.
No broker-sensitive artifacts tracked.
No live-order evidence.
No unaudited numeric report claims.
```

---

## 21. Appendix C — Final Report Completion Checklist

Before PDF export:

* [ ] All numeric placeholders resolved or explicitly left as placeholders.
* [ ] Every major claim appears in `result_claims_audit.md`.
* [ ] Every table appears in `table_inventory.md`.
* [ ] Every figure appears in `figure_inventory.md`.
* [ ] All figures have research-proxy or diagnostic captions.
* [ ] No account-return wording.
* [ ] No live-trading profitability wording.
* [ ] No true option-chain PnL wording.
* [ ] No causal cross-market wording.
* [ ] No true MSGARCH wording.
* [ ] Phase 11 is appendix-only.
* [ ] Phase 12 is skipped / future optional.
* [ ] Limitations are complete.
* [ ] Future work is complete.
* [ ] Reproducibility note is complete.
* [ ] PDF visually inspected after export.
