# Executive Summary

## Project

**Variance Risk Premium Decomposition and Regime-Conditional Harvesting: A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

This project studies the variance risk premium across the US and Indian equity-index volatility markets. It measures implied variance using VIX and India VIX proxies, estimates realised variance from daily OHLC data, builds VRP features, decomposes the premium across regimes, evaluates regime-conditioned research-proxy short-volatility exposure, and compares US and Indian regime behaviour.

The project is an empirical research and strategy-evaluation study. It is not a live trading system.

---

## Objective

The core research objective is to test whether regime-conditioned short-volatility exposure improves research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting.

---

## Evidence-Backed Summary

- VRP construction artifacts and metadata are available locally; metadata confirms the VIX/India VIX proxy convention, `rv_gk_22d_ann`, a 22-trading-day realised-variance horizon, and 252-day annualisation.
- HAR forecast and HAR no-lookahead audit artifacts are available; available HAR forecasts have `0` inspected violations of `rule_target_end_before_forecast_date`.
- Threshold, HMM, MAR, and MSVOL diagnostics are available. HMM and MAR no-lookahead audits pass for both markets, and MSVOL is diagnostic-only.
- Phase 9 signal diagnostics cover `7` US strategies and `7` India strategies. Forward/ex-post labels and diagnostic smoothed probabilities are marked as excluded from strategy use.
- Phase 10 provides seven-strategy research-proxy comparisons for both US and India. Annualised metrics are approximate because the proxy outcomes overlap.
- Phase 11 guard evidence reports `passed=true`, `violations=[]`, and inspected metadata reports `live_order_sent=false`; no broker orders were placed.
- Phase 13 alignment and no-lookahead tables are available, and logistic/lead-lag diagnostics are interpreted as predictive/statistical diagnostics only, not causal proof.

---

## Market Coverage

| Market | Underlying | Implied-volatility proxy |
|---|---|---|
| US | SPX / SPY | VIX |
| India | NIFTY 50 | India VIX |

VIX and India VIX are used as implied-volatility proxies. They are not variance swap quotes.

---

## Main Findings

1. The project constructs dual-market VRP proxy panels using inspected local metadata and summary tables.
2. HAR-RV provides a model-dependent, point-in-time forecast layer for prospective VRP construction.
3. The regime ladder gives interpretable threshold, HMM, and MAR state summaries, with MAR adding a reduced-form AR-aware layer.
4. The Phase 10 backtest is a vectorised research-proxy comparison, not an executable trading or account-return result.
5. Robustness artifacts cover cost sensitivity, subperiods, crisis windows, tail diagnostics, and tradable proxy detection.
6. Phase 13 cross-market diagnostics are descriptive or predictive/statistical only.
7. Phase 11 validates paper-signal readiness and live-order guard behaviour; no broker orders were placed.

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

| Deliverable | Path |
|---|---|
| Final report source | `reports/final/final_report.md` |
| Final report PDF | `reports/final/final_report.pdf` |
| Executive summary | `reports/final/executive_summary.md` |
| Presentation outline | `reports/final/presentation_outline.md` |
| Claims audit | `reports/final/result_claims_audit.md` |
| Selected artifacts | `reports/final/selected_artifacts.md` |
| Table inventory | `reports/final/table_inventory.md` |
| Figure inventory | `reports/final/figure_inventory.md` |
| Limitations | `reports/final/limitations.md` |
| Reproducibility note | `reports/final/reproducibility_note.md` |
| Future work | `reports/final/future_work.md` |
| Release checklist | `docs/release_checklist.md` |
| Submission package map | `docs/submission_package.md` |
