# Executive Summary

## Project

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

This project studies the variance risk premium across the US and Indian equity-index volatility markets. It measures implied variance using VIX and India VIX proxies, estimates realised variance from daily OHLC data, builds VRP features, decomposes the premium across regimes, evaluates regime-conditioned research-proxy short-volatility exposure, and compares US and Indian regime behaviour.

The project is an empirical research and strategy-evaluation study. It is not a live trading system.

---

## Objective

The core research objective is to test whether regime-conditioned short-volatility exposure improves research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting.

The project asks:

1. Can VRP be constructed consistently for both the US and Indian markets using public daily data?
2. Does HAR-RV forecasting provide a useful prospective realised-variance layer?
3. Do threshold, Gaussian HMM, and Markov autoregression models produce interpretable volatility regimes?
4. Do regime-conditioned exposure intentions improve research-proxy backtest characteristics relative to the unconditional benchmark?
5. Are the results robust across cost assumptions, subperiods, and stress windows?
6. Do US and Indian VRP/regime diagnostics show descriptive or predictive cross-market relationships?
7. Can the final signal be represented in a guarded paper-signal format without broker execution?

---

## Pipeline

The completed research pipeline is:

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

---

## Market Coverage

| Market | Underlying | Implied-volatility proxy |
| ------ | ---------- | ------------------------ |
| US     | SPX / SPY  | VIX                      |
| India  | NIFTY 50   | India VIX                |

VIX and India VIX are used as implied-volatility proxies. They are not variance swap quotes.

---

## Methodology

The project uses a layered methodology.

| Layer                   | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| Data layer              | Public daily data ingestion and validation                           |
| Realised variance layer | Daily OHLC-based realised variance estimators                        |
| Implied variance layer  | VIX and India VIX converted into implied variance proxies            |
| VRP layer               | Ex-post and HAR-based VRP construction                               |
| Forecasting layer       | HAR-RV forecasting under point-in-time constraints                   |
| Regime layer            | Threshold, Gaussian HMM, Markov autoregression, and MSVOL robustness |
| Strategy layer          | Regime-conditioned exposure intentions                               |
| Backtest layer          | Vectorised research-proxy backtest and robustness                    |
| Cross-market layer      | US-India same-date and lagged predictive diagnostics                 |
| Operational appendix    | IBKR paper-signal readiness with live-order guards                   |

---

## Main Findings

Final numeric findings must be filled only after inspecting local result tables.

### 1. VRP construction

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/vrp_summary.csv]
```

Allowed wording:

```text
The project constructs aligned US and India VRP panels using VIX/India VIX implied-variance proxies and daily OHLC realised-variance proxies.
```

### 2. HAR-RV forecasting

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/har_forecast_accuracy.csv]
```

Allowed wording:

```text
The HAR-RV layer provides a model-dependent, point-in-time realised-variance forecast for prospective VRP construction.
```

### 3. Regime modelling

```text
[INSERT VERIFIED SUMMARY FROM PHASE 5/6/7 REGIME SUMMARY TABLES]
```

Allowed wording:

```text
The regime ladder gives interpretable market-state classifications, with Markov autoregression adding an AR-aware extension beyond the Gaussian HMM layer.
```

### 4. Research-proxy backtest

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/phase_10/backtest_summary.csv]
```

Allowed wording:

```text
In the vectorised research-proxy backtest, selected regime-conditioned variants show [INSERT VERIFIED RELATION] relative to the unconditional benchmark in the tested sample.
```

Forbidden wording:

```text
The strategy is profitable in live trading.
```

### 5. Robustness

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/phase_10/robustness_cost_sensitivity.csv AND reports/tables/phase_10/robustness_subperiods.csv]
```

Allowed wording:

```text
Robustness checks describe how research-proxy results vary across cost assumptions and subperiods.
```

### 6. Cross-market diagnostics

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/phase_13/logistic_model_comparison.csv AND reports/tables/phase_13/lead_lag_table.csv]
```

Allowed wording:

```text
Cross-market diagnostics show [INSERT VERIFIED ASSOCIATION] as statistical/predictive evidence, not causal proof.
```

### 7. Paper-signal readiness

```text
[INSERT VERIFIED SUMMARY FROM reports/tables/phase_11/live_order_guard_report.json]
```

Allowed wording:

```text
The Phase 11 appendix validates paper-signal readiness and live-order guard behaviour; no broker orders were placed.
```

---

## What the Project Demonstrates

The project demonstrates:

1. A complete public-data volatility research pipeline.
2. Careful separation between realised outcomes and tradable features.
3. Strict no-lookahead discipline for forecasting, regimes, signals, and backtests.
4. A regime model ladder from simple thresholds to latent-state and AR-aware models.
5. A vectorised research-proxy backtest with robustness checks.
6. Cross-market US-India analysis with explicit non-causal wording.
7. A paper-signal readiness layer with live-order guards.
8. A final release package with claims audit, artifact inventory, limitations, and reproducibility notes.

---

## What the Project Does Not Claim

The project does not claim:

```text
live-trading profitability
true option-chain PnL
account returns
causal US-to-India transmission
true R MSGARCH implementation
broker order execution
Phase 12 implementation
```

---

## Key Caveats

1. VIX and India VIX are implied-volatility proxies, not variance swap quotes.
2. Daily OHLC realised variance estimators are proxies for true realised variance.
3. A 22-trading-day horizon approximates one month but is not equivalent to 30 calendar days.
4. Forward ex-post VRP labels are evaluation outcomes, not tradable features.
5. HAR forecasts are model-dependent.
6. Gaussian HMM does not directly model observed-series autocorrelation.
7. Markov autoregression remains a reduced-form regime model.
8. MSVOL is Python-only volatility-regime robustness, not true R MSGARCH.
9. Strategy outputs are exposure intentions, not broker orders.
10. Phase 10 backtest outputs are research-layer proxy results, not executable account returns.
11. Phase 11 did not place broker orders.
12. Phase 12 was skipped and remains future optional work.
13. Phase 13 cross-market diagnostics are statistical/predictive diagnostics, not causal proof.
14. Generated data panels remain local and are not committed to GitHub.

---

## Final Deliverables

| Deliverable            | Path                                    |
| ---------------------- | --------------------------------------- |
| Final report source    | `reports/final/final_report.md`         |
| Final report PDF       | `reports/final/final_report.pdf`        |
| Executive summary      | `reports/final/executive_summary.md`    |
| Presentation outline   | `reports/final/presentation_outline.md` |
| Claims audit           | `reports/final/result_claims_audit.md`  |
| Selected artifacts     | `reports/final/selected_artifacts.md`   |
| Table inventory        | `reports/final/table_inventory.md`      |
| Figure inventory       | `reports/final/figure_inventory.md`     |
| Limitations            | `reports/final/limitations.md`          |
| Reproducibility note   | `reports/final/reproducibility_note.md` |
| Future work            | `reports/final/future_work.md`          |
| Release checklist      | `docs/release_checklist.md`             |
| Submission package map | `docs/submission_package.md`            |
