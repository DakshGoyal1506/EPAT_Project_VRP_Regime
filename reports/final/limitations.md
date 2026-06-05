# Final Limitations

This document is the final-report version of the project limitations. It is curated from `docs/known_limitations.md` and should be read together with the full project documentation.

The limitations below are not minor disclaimers. They define the boundary of what this project can and cannot claim.

---

## 1. Data Source Limitations

The project uses public daily market data and therefore inherits public-source limitations.

1. Public vendors can change file formats, ticker coverage, historical depth, download behaviour, or revision policies.
2. Yahoo Finance data via `yfinance` is a convenient access layer, not an official exchange record.
3. NSE scripted access may be blocked, rate-limited, or format-shifted; manual CSV override is supported by design.
4. CBOE, FRED, Yahoo, NSE, and other sources may differ because of timestamp, close definition, methodology, and revision differences.
5. Official/manual source downloads must be documented locally when used.
6. Raw market data is not committed to GitHub. Reviewers inspect reproducibility instructions, audit summaries, and selected artifacts instead.

---

## 2. Calendar and Alignment Limitations

The US and Indian markets have different trading calendars, holidays, trading hours, and local closing times.

1. US and India dates are not interchangeable.
2. Same-date cross-market comparisons are descriptive only.
3. Predictive cross-market tests must use explicitly lagged information.
4. Holiday gaps can create stale lagged observations.
5. Daily date-level storage is used; intraday timezone mechanics are outside the core public-data research scope.
6. Calendar alignment must be interpreted as a modelling choice, not as an exact reconstruction of real-time global information flow.

---

## 3. Realised Variance Limitations

The project estimates realised variance from daily OHLC data.

1. Daily OHLC realised variance estimators are proxies for true realised variance.
2. Daily OHLC data does not observe full intraday price paths.
3. Range-based estimators are sensitive to bad high/low data.
4. Close-to-close volatility ignores intraday range.
5. Garman-Klass is used as the primary realised variance proxy, but no OHLC estimator is universally true.
6. The selected realised variance estimator must remain consistent when compared against annualised implied variance.
7. A 22-trading-day window approximates a one-month trading horizon but is not identical to 30 calendar days.

---

## 4. Implied Variance and VRP Limitations

The project uses VIX and India VIX as implied-volatility index proxies.

1. VIX and India VIX are implied-volatility proxies, not direct variance swap quotes.
2. Squared VIX and squared India VIX are model-free implied variance approximations, not exact tradable variance swap rates.
3. Index methodology, option universe, settlement mechanics, and local market microstructure differ across the US and India.
4. Ex-post forward VRP labels use future realised variance and are therefore outcomes only.
5. Forward realised variance labels must not be used as tradable signals.
6. HAR-based prospective VRP is model-dependent and relies on forecasts available at the signal timestamp.
7. VRP construction is suitable for public-data empirical research, not direct account-level option PnL reconstruction.

---

## 5. Forecasting Limitations

The HAR-RV layer is a forecasting model, not a trading guarantee.

1. HAR-RV forecasts depend on upstream realised variance estimator choices.
2. Forecasts are point estimates and can be wrong during regime shifts.
3. Expanding or rolling training windows reduce lookahead risk but do not eliminate model risk.
4. Forecast quality can be sample-sensitive.
5. GPU acceleration is optional and not required for conceptual reproducibility.
6. HAC coefficient inference and forecast production settings can affect diagnostics.
7. HAR-based VRP should be interpreted as a model-based feature, not as known future carry.

---

## 6. Regime Model Limitations

The project uses several regime models, each with different assumptions.

### Threshold regimes

1. Threshold regimes are deterministic baseline labels.
2. Threshold choices affect state counts and interpretation.
3. Calm states can be sparse if the rule requires several conservative conditions simultaneously.
4. Crisis-window diagnostics are reporting-only and must not be used as hidden calibration labels.

### Gaussian HMM

1. Gaussian HMM emissions do not directly model autoregression in the observed VRP/RV series.
2. HMM states are latent statistical regimes, not observed ground truth.
3. Economic state labels are interpretations, not facts.
4. State-count selection can be sample-sensitive.
5. Full-sample smoothed probabilities are diagnostic only.
6. Backtest-facing HMM signals must use filtered probabilities available at time `t`.

### Markov autoregression

1. Markov autoregression addresses observed-series autocorrelation more directly than Gaussian HMM.
2. It remains a reduced-form regime model.
3. Fitting can be numerically sensitive.
4. State labels remain economic interpretations.
5. AR-aware regimes do not prove structural market causality.

### MSVOL robustness

1. Phase 8 MSVOL is a Python-only Markov-switching volatility robustness proxy.
2. It is not true R `MSGARCH`.
3. True R MSGARCH remains optional/future work.
4. MSVOL outputs are diagnostic-only.
5. MSVOL outputs are not used for Phase 9 strategy construction, Phase 10 backtesting, VaR/ES, or Phase 13 cross-market analysis.

---

## 7. Strategy Signal Limitations

The strategy layer produces exposure intentions.

1. Strategy outputs are not broker orders.
2. Strategy outputs do not specify option contracts, strikes, expiries, greeks, margin, borrow, or account sizing.
3. Exposure rules depend on model outputs and carry filters, which can be wrong.
4. Regime-conditioned suppression can reduce exposure during some stress periods but cannot guarantee drawdown avoidance.
5. Forward ex-post VRP labels are never tradable features.
6. Signal timing must use only information available at the decision timestamp.

---

## 8. Backtest Limitations

Phase 10 is a vectorised research-proxy backtest.

1. Phase 10 returns are research-layer proxy units, not executable account returns.
2. Cumulative curves are additive proxy curves, not account equity curves.
3. The backtest does not model true option-chain PnL.
4. The backtest does not model instrument-level execution.
5. There is no true margin, capital, contract multiplier, strike selection, expiry selection, delta hedge, assignment, or liquidity model.
6. Transaction costs are assumptions in proxy units unless tied to executable instrument data.
7. Overlapping 22-day outcome labels make annualised metrics approximate.
8. Crisis-window diagnostics are historical descriptions, not guarantees of future crisis protection.
9. Good research-proxy results do not imply live-trading profitability.
10. Backtest outputs must be described as empirical evidence, not as deployable performance.

---

## 9. Robustness Limitations

Robustness checks test sensitivity, not truth.

1. Cost sensitivity evaluates assumed proxy costs, not all real execution frictions.
2. Subperiod robustness depends on sample segmentation.
3. Crisis-window checks depend on selected historical periods.
4. Tail diagnostics are sample-dependent.
5. Robustness results cannot prove future stability.
6. Model comparison can show relative behaviour in-sample or out-of-sample, but not universal superiority.

---

## 10. Cross-Market Analysis Limitations

Phase 13 is analysis-only.

1. Same-date US-India diagnostics are descriptive only.
2. Lagged-US diagnostics are predictive/statistical tests, not causal proof.
3. Granger-style diagnostics are lead-lag diagnostics, not structural causality.
4. Logistic incremental-signal tests show tested-sample predictive association only.
5. Indian market holidays can create stale lagged US observations.
6. Daily close-level alignment does not model intraday information transmission.
7. The India overlay is analysis-only and is not part of the locked Phase 9 strategy universe.
8. Overlay results do not constitute live-trading evidence, strategy implementation, or execution evidence.

---

## 11. Broker / Paper-Signal Limitations

Phase 11 is a paper-signal readiness appendix.

1. iBridgePy / IBKR is optional and not required for core research reproducibility.
2. Phase 11 did not place broker orders.
3. Broker integration is paper/signal-only by default.
4. Live orders are disabled by config and policy.
5. Broker account identifiers remain local.
6. Broker cache, logs, runtime metadata, and account-sensitive outputs are local-only unless redacted.
7. Instrument availability and permissions must be verified outside the research pipeline.
8. The project does not infer live option sizing from research-proxy backtests.

---

## 12. Phase 12 Limitation

Phase 12 was intentionally skipped.

Correct wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

Phase 12 should not be described as implemented, partially implemented, or validated.

---

## 13. Final Report and Artifact Limitations

The final report is based on tracked code, documentation, and local generated artifacts.

1. Generated data panels remain local and are not committed to GitHub.
2. Full backtest panels remain local.
3. Full strategy signal panels remain local.
4. Full cross-market panels remain local.
5. Broker runtime outputs remain local or redacted.
6. Numeric findings should be inserted only after inspecting local result tables.
7. Every major conclusion must map to `reports/final/result_claims_audit.md`.
8. Every final table must map to `reports/final/table_inventory.md`.
9. Every final figure must map to `reports/final/figure_inventory.md`.
10. The PDF must be generated from the audited Markdown report source.

---

## 14. Forbidden Overclaims

The final report must not claim:

```text
live-trading profitability
true option-chain PnL
account returns
causal US-to-India transmission
true R MSGARCH implementation
broker order execution
Phase 12 implementation
guaranteed drawdown reduction
future crisis protection
production deployment readiness
```

---

## 15. Approved Wording

Use these terms instead:

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
drawdown behaviour in the tested proxy sample
```
