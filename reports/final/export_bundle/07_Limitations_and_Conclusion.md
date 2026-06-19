# 7. Methodological Limitations & Conclusion

## Core Limitations
1. **Implied-Volatility Proxies:** VIX and India VIX are implied-volatility proxies, not true variance swap quotes. Squared VIX is a model-free implied variance approximation, but true swap rates embed distinct settlement dynamics.
2. **Realised Variance:** Daily OHLC realised variance estimators are proxies. They do not observe full intraday tick-level price paths.
3. **Forecasting Restrictions:** HAR forecasts are model-dependent and constrained by the historical length and weighting of the training-windows.
4. **Regime Models:** Gaussian HMM and MAR use statistical state labels that act as economic interpretations rather than observed ground truth. 
5. **Research-Proxy Backtests:** The Phase 10 backtest uses research-layer proxy units. Cumulative curves are additive proxy curves, not executable account equity curves. The backtest does not model true option-chain PnL, margin, liquidity, bid/ask spread, or instrument-level execution.
6. **Cross-Market Diagnostics:** Phase 13 diagnostics are statistical/predictive diagnostics only. They do not constitute causal transmission proof.

## Conclusion and Future Scope
The dual-market VRP empirical study successfully demonstrates that regime-conditioned exposure strategies structurally improve risk-adjusted proxy returns over unconditional short-volatility harvesting. By deploying Threshold, HMM, and Markov Autoregressive models, the left-tail variance risks inherent in equity index option premia can be systematically managed.

**Future Scope:**
- **True Option-Chain PnL:** Expanding the backtest to utilize historical option chains to construct executable option portfolios, selecting precise strikes/expiries, modelling bid/ask spreads, and computing true option-level PnL.
- **Variance Swap Data:** Utilizing actual institutional variance swap quotes to compare against the VIX-squared proxy variance.
- **True R MSGARCH:** Implementing formal Markov-switching GARCH using R and comparing the states with current MSVOL robustness proxies.
- **IBKR Paper Execution Adapter:** Building out a paper execution adapter (skipped Phase 12) to convert paper-signal intents into API-driven paper-order previews to validate live routing boundaries.
- **Portfolio and Capital Allocation:** Adding explicit margin usage limits, maximum drawdown cutoffs, and cross-market risk aggregation to escalate the research-proxy into a full deployable portfolio construction model.

*Note: Phase 11 serves as an operational readiness appendix. It converts research signals into guarded paper-signal artifacts and validates that live-order pathways remain explicitly blocked. No broker orders were placed during this research project.*
