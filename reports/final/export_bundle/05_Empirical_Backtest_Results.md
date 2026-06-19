# 5. Empirical Backtest Results

The strategy layer converts regime and carry information into next-session exposure intentions. The vectorised research-proxy backtest compares unconditional short-volatility harvesting against regime-conditioned rules.

## Strategy Performance Profiles

[INSERT IMAGE: equity_curves_common_start_india.png]
**Mathematical Interpretation:**
The common-start equity curves represent additive research-proxy return trajectories $P_T = \sum_{t=1}^T w_t \cdot VRP_t$, where $w_t \in [-1, 0]$ is the regime-conditioned exposure intent. By standardizing the inception date, we observe how dynamically truncating $w_t \to 0$ during Indian market stress events structurally preserves accumulated proxy capital compared to the unconditional baseline.

[INSERT IMAGE: equity_curves_common_start_us.png]
**Mathematical Interpretation:**
For the US market, the regime-aware models (e.g., MAR and Gaussian HMM) dynamically limit exposure during identified stress periods like 2008 and 2020. The compounded effects of avoiding severe left-tail negative $VRP_t$ realizations result in superior long-term risk-adjusted capital paths despite sacrificing carry during calmer periods.

[INSERT IMAGE: equity_curves_india.png]
**Mathematical Interpretation:**
Full-sample trajectory for the Indian market confirming the regime-conditioned models fundamentally alter the return distributions, resulting in much smoother paths with mitigated volatility clustering in the equity curve.

[INSERT IMAGE: equity_curves_us.png]
**Mathematical Interpretation:**
Full-sample trajectory for the US market illustrating the compounding of proxy return units over decades.

## Drawdown and Risk Analysis

[INSERT IMAGE: drawdowns_india.png]
**Mathematical Interpretation:**
The drawdown curves explicitly track the underwater magnitude: $D_t = \max_{\tau \in (0, t)} P_\tau - P_t$. For the Indian market, the unconditional baseline suffers extreme capitulations exceeding a -6.0 proxy unit drawdown. Conversely, the HMM and MAR models actively constrain the maximum drawdown, acting mathematically as a stop-loss function derived from latent state probabilities.

[INSERT IMAGE: drawdowns_us.png]
**Mathematical Interpretation:**
In the US market, the effectiveness of the regime suppression is stark. The unconditional strategy yields massive drawdowns during the GFC and Volmageddon, whereas the Markov autoregressive models explicitly truncate exposure, compressing $D_t$ and resulting in vastly superior Calmar and Sortino ratios.

[INSERT IMAGE: return_distribution_india.png]
**Mathematical Interpretation:**
The return distribution histogram visualizes the alteration of the $VRP_t$ density function. Regime conditioning physically truncates the extreme left tail of the return distribution. By avoiding these negative realizations, the semi-variance is minimized.

[INSERT IMAGE: return_distribution_us.png]
**Mathematical Interpretation:**
For the US, the histogram confirms the same structural improvement. Unconditional short-volatility yields a long, fat left tail of catastrophic losses. The regime-filtered strategies concentrate returns near zero (from periods out of the market) while preserving the high-frequency positive carry realizations of the calm regime.
