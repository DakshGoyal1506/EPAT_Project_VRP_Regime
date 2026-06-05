# Future Work

This document lists extensions that are intentionally outside the current EPAT submission scope.

The current project is complete as a public-data empirical research pipeline, vectorised research-proxy backtest, cross-market analysis, and paper-signal readiness package. The items below should not be implied as implemented.

---

## 1. True Option-Chain PnL

The current backtest is a research-layer VRP proxy backtest. It does not reconstruct true option-chain PnL.

Future extension:

```text
Use historical option chains to construct executable option portfolios.
Select strikes and expiries.
Model bid/ask spreads.
Model liquidity and open interest filters.
Track delta, gamma, vega, theta, and margin.
Compute true option-level PnL.
```

This would convert the project from VRP proxy research to instrument-level options backtesting.

---

## 2. Variance Swap or Volatility Swap Data

The current project uses VIX and India VIX as implied-volatility proxies.

Future extension:

```text
Use actual variance swap quotes where available.
Compare VIX-squared proxy variance with variance swap rates.
Evaluate variance swap PnL directly.
Study variance swap term structure.
```

This would reduce proxy error but requires data that is often private, expensive, or unavailable for the Indian market.

---

## 3. Intraday Realised Variance

The current realised variance layer uses daily OHLC data.

Future extension:

```text
Use intraday bars or tick data.
Estimate realised variance from high-frequency returns.
Handle market microstructure noise.
Handle bid/ask bounce.
Separate overnight and intraday variance.
Compare daily OHLC estimators against intraday realised variance.
```

This would improve realised variance measurement but increase data-cleaning and computational complexity.

---

## 4. True R MSGARCH Implementation

Phase 8 uses Python-only MSVOL robustness. It is not true R MSGARCH.

Future extension:

```text
Install and configure the R MSGARCH package.
Export return series to R.
Fit Markov-switching GARCH models.
Import fitted states and volatility forecasts.
Compare MSVOL, MSGARCH, HMM, and MAR regimes.
Document differences between return-volatility and VRP-regime models.
```

This should remain an appendix unless it materially changes core conclusions.

---

## 5. IBKR Paper Execution Adapter

Phase 12 was intentionally skipped and left as future optional work.

Future extension:

```text
Build an IBKR/iBridgePy paper execution adapter.
Convert paper-signal intents into paper-order previews.
Add contract lookup and permission checks.
Add dry-run order construction.
Add strict live-order blocking.
Log rejected, blocked, and previewed paper orders.
Validate no accidental live routing.
```

Correct final wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

This extension must not be described as completed in the current report.

---

## 6. Production-Grade Broker Deployment

The current project is not a production trading system.

Future extension:

```text
Add deployment environment separation.
Add secure credential handling.
Add broker permission checks.
Add portfolio state reconciliation.
Add risk limit monitoring.
Add kill-switch logic.
Add execution logs.
Add alerting.
Add post-trade reconciliation.
```

This would require a separate engineering and compliance scope.

---

## 7. Tradable Instrument Mapping

The current strategy outputs exposure intentions.

Future extension:

```text
Map exposure intentions to actual instruments.
Evaluate SPX options, SPY options, VIX futures, NIFTY options, India VIX-related instruments if available.
Model contract size.
Model roll rules.
Model option expiry selection.
Model slippage and liquidity.
Model margin and capital requirements.
```

This is necessary before any claim about executable account returns.

---

## 8. Portfolio and Capital Allocation Layer

The current project does not define account capital, margin allocation, or portfolio-level risk budgeting.

Future extension:

```text
Define initial capital.
Define volatility target.
Define margin usage limits.
Define max drawdown cutoffs.
Define per-market allocation.
Define cross-market risk aggregation.
Define portfolio-level stop or de-risking rules.
```

This would turn the research-proxy evaluation into a portfolio construction exercise.

---

## 9. Stronger Statistical Inference

The current project includes diagnostics and robustness checks, but overlapping labels and regime dependence complicate inference.

Future extension:

```text
Use block bootstrap inference.
Use Newey-West or HAC adjustments where appropriate.
Use reality-check or multiple-testing correction.
Evaluate model confidence intervals.
Evaluate uncertainty in regime labels.
Evaluate parameter stability across samples.
```

This would strengthen formal statistical claims.

---

## 10. Additional Markets

The project currently studies the US and India.

Future extension:

```text
Add Europe volatility index markets.
Add Japan volatility index markets.
Add Korea or Hong Kong volatility markets.
Compare market maturity, liquidity, and regime synchronization.
Study global volatility spillovers with explicit timezone handling.
```

This would test whether findings generalize beyond the dual-market design.

---

## 11. Improved Cross-Market Design

Phase 13 uses daily close-level cross-market diagnostics.

Future extension:

```text
Use intraday timestamps.
Model timezone-aware information availability.
Separate global overnight moves from local trading-session effects.
Use event-study methods around global shocks.
Evaluate whether US information remains predictive after controlling for global risk factors.
```

This would improve cross-market interpretation but still would not automatically prove causality.

---

## 12. Alternative Forecast Models

The project uses HAR-RV as the main forecast layer.

Future extension:

```text
Compare HAR-RV with GARCH-family forecasts.
Compare tree-based forecasts.
Compare regularized linear models.
Compare regime-specific HAR models.
Compare forecast combinations.
Evaluate forecast degradation across stress regimes.
```

Any additional forecast model must preserve point-in-time training and no-lookahead constraints.

---

## 13. Alternative Regime Models

The current regime ladder includes threshold regimes, Gaussian HMM, MAR, and MSVOL robustness.

Future extension:

```text
Add Bayesian HMM.
Add time-varying transition probabilities.
Add hidden semi-Markov models.
Add dynamic factor regimes.
Add multivariate regime-switching models.
Add AR-HMM through Dynamax/JAX if explicitly re-scoped.
```

These should be treated as model-risk extensions, not guaranteed improvements.

---

## 14. Report Automation

The current final report is Markdown-first with PDF export.

Future extension:

```text
Build a report-generation script.
Automatically pull selected metrics from audited CSV/JSON files.
Generate final tables from templates.
Validate that all placeholders are resolved.
Validate that every claim maps to the claims audit.
Generate Markdown, DOCX, and PDF outputs from one source.
```

This would reduce manual report risk in future iterations.

---

## 15. Public Release Hardening

Future public-release work:

```text
Add CI workflow.
Add artifact-size checks.
Add secret scanning.
Add notebook-output clearing checks.
Add markdown linting.
Add reproducibility smoke test.
Add release tag.
Add final ZIP excluding local-only artifacts.
```

This would make the repo easier to distribute and review.

---

## 16. Explicit Non-Goals for Current Submission

The following are not current deliverables:

```text
live trading
paper order execution
true option-chain backtest
true account returns
production broker deployment
causal cross-market proof
true R MSGARCH implementation
full generated data upload
```

These should remain future work unless explicitly re-scoped after EPAT submission.
