# Variance Risk Premium Decomposition and Regime-Conditional Harvesting
**A Dual-Market Empirical Study across SPX/VIX in the US and NIFTY/India VIX in India**

---

## Section 1: Abstract & Executive Summary

The variance risk premium (VRP) is the empirical spread between implied variance and subsequently realised variance. In equity-index markets, implied volatility indices such as VIX and India VIX often stand above subsequent realised volatility, but unconditional short-volatility exposure can suffer severe losses during stress periods. 

This project studies whether the variance risk premium harvesting can be made more defensible by conditioning exposure on volatility regimes. It builds a reproducible dual-market research pipeline for the US and India. The core research objective is to test whether regime-conditioned short-volatility exposure improves research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting.

The research design has three central components:
1. **VRP measurement:** estimate realised variance from daily OHLC data and compare it with implied variance proxies derived from VIX and India VIX.
2. **Regime decomposition:** classify market states using threshold regimes, Gaussian HMM, Markov autoregression, and MSVOL robustness diagnostics.
3. **Research-proxy strategy evaluation:** compare unconditional short-volatility harvesting with regime-conditioned exposure intentions in a vectorised backtest.

---

## Section 2: Core Methodology & Realised Variance Estimation

Realised variance is estimated from daily OHLC data. The project includes multiple estimators, with Garman-Klass 22-trading-day annualised realised variance used as the primary proxy. VIX and India VIX are converted into implied variance proxies by squaring the volatility index after percentage scaling. 

Conceptually:
`implied_variance_proxy = (implied_volatility_index / 100)^2`

The VRP is constructed by comparing the implied variance proxy against the realised variance. The backward VRP uses lagged realised variance, while the forward ex-post VRP label is retained strictly as an evaluation outcome, not a tradable feature.

### Realised Variance Estimators Visualization

![Visual comparison of different realized variance estimators for India](../figures/rv_estimators_india.png)
*Figure 1: Daily OHLC realised-variance estimator comparison, India. The estimators highlight the variance bounds using close-to-close, Parkinson, Garman-Klass, and Rogers-Satchell methods.*

![Visual comparison of different realized variance estimators for the US](../figures/rv_estimators_us.png)
*Figure 2: Daily OHLC realised-variance estimator comparison, US.*

---

## Section 3: Predictive Volatility Modeling

The HAR-RV (Heterogeneous Autoregressive model of Realised Volatility) layer estimates prospective realised variance using lagged realised-variance features. Its role is to provide a model-based expected realised variance input for prospective VRP construction under point-in-time constraints.

### HAR-RV Forecasts and Residuals

![Plot representing HAR-RV forecast for the Indian market](../figures/har_forecast_india.png)

![Plot representing HAR-RV forecast for the US market](../figures/har_forecast_us.png)

![Plot of HAR-RV model residuals for the Indian market](../figures/har_residuals_india.png)

![Plot of HAR-RV model residuals for the US market](../figures/har_residuals_us.png)

**Mathematical Interpretation:**
The HAR-RV model constructs a prospective realized variance forecast $RV_{t, t+22}$ using lagged daily, weekly, and monthly realized variance components $RV^{(d)}_t$, $RV^{(w)}_t$, $RV^{(m)}_t$. The forecast plot demonstrates that while the model captures the low-frequency volatility persistence and mean-reversion characteristic of the market, the residuals (forecast errors) $\epsilon_t = RV_{t, t+22} - \widehat{RV}_{t, t+22}$ often exhibit heteroskedasticity and fat tails during volatility spikes. This highlights the limitations of linear autoregressive structures during sudden regime shifts.

![Plot of HAR-based prospective VRP for India](../figures/har_vrp_india.png)

![Plot of HAR-based prospective VRP for the US](../figures/har_vrp_us.png)

---

## Section 4: Regime Detection Architectures

The regime modelling ladder addresses the state classification of the market using progressively sophisticated models:
- **Threshold Regimes:** Provide a simple, deterministic baseline.
- **Gaussian HMM:** Estimates latent regimes from observed features using filtered probabilities.
- **Markov Autoregression (MAR):** Extends the regime ladder by allowing state-dependent autoregressive dynamics in the observed series, addressing autocorrelation constraints in standard HMMs.

### Regime Visualization and VRP Distributions

![Overall visualization of threshold regimes for India](../figures/threshold_regimes_india.png)

![Overall visualization of threshold regimes for the US](../figures/threshold_regimes_us.png)

![Plot of states identified by the threshold component model for India](../figures/threshold_component_states_india.png)

![Plot of states identified by the threshold component model for the US](../figures/threshold_component_states_us.png)

![Boxplot distribution of VRP across different threshold regimes for India](../figures/threshold_regime_vrp_boxplots_india.png)

![Boxplot distribution of VRP across different threshold regimes for the US](../figures/threshold_regime_vrp_boxplots_us.png)

**Mathematical Interpretation:**
The boxplots visualize the empirical distribution of the Variance Risk Premium ($VRP_t = IV_{t, t+22} - \mathbb{E}_t[RV_{t, t+22}]$) conditioned on the deterministic threshold regimes. We observe that during 'stress' regimes, the VRP distribution exhibits significantly higher dispersion and a thicker right tail compared to 'calm' and 'transition' regimes. This mathematically indicates that while volatility risk premia expand during turbulent periods to compensate for jump and crash risks, the variance of the premium itself also significantly increases, reducing risk-adjusted predictability.

---

## Section 5: Systematic Harvesting Strategies & Empirical Backtest Results

The strategy layer converts regime and carry information into next-session exposure intentions. The vectorised research-proxy backtest compares unconditional short-volatility harvesting against regime-conditioned rules. 

### Strategy Performance Profiles

![Strategy equity curves starting from a common date for India](../figures/phase_10/equity_curves_common_start_india.png)

![Strategy equity curves starting from a common date for the US](../figures/phase_10/equity_curves_common_start_us.png)

![Overall strategy equity curves for India](../figures/phase_10/equity_curves_india.png)

![Overall strategy equity curves for the US](../figures/phase_10/equity_curves_us.png)

![Drawdown chart of the backtested strategy for India](../figures/phase_10/drawdowns_india.png)

![Drawdown chart of the backtested strategy for the US](../figures/phase_10/drawdowns_us.png)

![Strategy return distribution histogram for India](../figures/phase_10/return_distribution_india.png)

![Strategy return distribution histogram for the US](../figures/phase_10/return_distribution_us.png)

**Mathematical Interpretation:**
The full-sample and common-start equity curves represent additive research-proxy return trajectories $P_T = \sum_{t=1}^T w_t \cdot VRP_t$, where $w_t \in [-1, 0]$ is the regime-conditioned exposure intent. The regime-aware models (e.g., MAR and Gaussian HMM) dynamically truncate exposure during identified stress periods. This is mathematically evident in the drawdown curves where the maximum drawdown $\max_{\tau \in (0, T)} (\max_{t \in (0, \tau)} P_t - P_\tau)$ is significantly constrained compared to the unconditional short-volatility baseline. The return distribution histograms further show a truncation of the left tail (negative VRP realizations) for the regime-conditioned strategies, improving the Sortino ratio by explicitly minimizing downside semi-variance.

---

## Section 6: Cross-Market Dynamics & Lead-Lag Analysis

Phase 13 adds an analysis-only cross-market layer to study same-date co-movement, lagged-US predictive diagnostics, and an analysis-only India overlay.

### US vs. India Cross-Market Diagnostics

![Comparative Variance Risk Premium plot for US and India](../figures/phase_13/us_india_vrp.png)

![Comparative stress probabilities between US and India](../figures/phase_13/us_india_stress_prob.png)

![Plot showing cross-market lead-lag stress dynamics (US lagging/leading India)](../figures/phase_13/lagged_us_vs_india_stress.png)

![Equity curves showing the Indian market overlaid for comparison](../figures/phase_13/india_overlay_equity_curves.png)

![Exposure overlay for the Indian market comparison](../figures/phase_13/india_overlay_exposure.png)

**Mathematical Interpretation:**
The cross-market diagnostics assess whether stress state probabilities in the US $\mathbb{P}(S^{US}_t = \text{stress})$ hold predictive or descriptive association with Indian market stress $\mathbb{P}(S^{IN}_{t+k} = \text{stress})$. The lagged scatter plots and logistic model overlays demonstrate a statistically significant but non-deterministic lead-lag relationship where US volatility shocks often precede Indian volatility expansion. The comparative VRP magnitude plots show that while the base premium exists in both markets, the Indian VRP frequently commands a structurally higher baseline, reflecting differing market microstructure, liquidity premia, and domestic retail composition.

---

## Section 7: Methodological Limitations & Future Scope

### Limitations
1. VIX and India VIX are implied-volatility proxies, not true variance swap quotes.
2. Daily OHLC realised variance estimators are proxies; they do not observe full intraday price paths.
3. HAR forecasts are model-dependent and constrained by training-window choices.
4. Regime Models (Gaussian HMM, MAR) use statistical state labels that act as economic interpretations rather than observed ground truth. Phase 8 MSVOL is Python-only robustness, not true R MSGARCH.
5. The Phase 10 backtest uses research-layer proxy units. Cumulative curves are additive proxy curves, not executable account equity curves. The backtest does not model true option-chain PnL, margin, or instrument-level execution.
6. Phase 13 diagnostics are statistical/predictive diagnostics, not causal transmission proof.

### Future Scope
1. **True Option-Chain PnL:** Use historical option chains to construct executable option portfolios, selecting strikes/expiries, modelling bid/ask spreads, and computing true option-level PnL.
2. **Variance Swap Data:** Utilize actual variance swap quotes to compare against VIX-squared proxy variance.
3. **True R MSGARCH:** Implement formal Markov-switching GARCH using R and compare with current MSVOL robustness proxies.
4. **IBKR Paper Execution Adapter:** Build a paper execution adapter (skipped Phase 12) to convert paper-signal intents into paper-order previews and validate live routing boundaries.
5. **Portfolio and Capital Allocation:** Add margin limits, max drawdown cutoffs, and cross-market risk aggregation to turn the research-proxy into a full portfolio construction model.
