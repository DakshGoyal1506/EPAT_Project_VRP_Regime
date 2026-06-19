# 6. Cross-Market Dynamics & Lead-Lag Analysis

Phase 13 adds an analysis-only cross-market layer to study same-date co-movement, lagged-US predictive diagnostics, and an analysis-only India overlay.

## Lead-Lag and Stress Probabilities

[INSERT IMAGE: us_india_vrp.png]
**Mathematical Interpretation:**
The comparative Variance Risk Premium magnitude plot overlays the harvested premiums. While the base premium exists in both markets, the Indian VRP frequently commands a structurally higher baseline ($\mathbb{E}[VRP^{IN}_t] > \mathbb{E}[VRP^{US}_t]$). This reflects differing market microstructure, illiquidity premia, and domestic retail participation, leading to a persistently higher risk-neutral expectation of variance relative to realization.

[INSERT IMAGE: us_india_stress_prob.png]
**Mathematical Interpretation:**
This plot overlays the filtered stress probabilities $\mathbb{P}(S^{US}_t = \text{stress})$ and $\mathbb{P}(S^{IN}_t = \text{stress})$ from the Gaussian HMMs. The temporal clustering of state vectors indicates systemic global volatility cycles, but local idiosyncrasies (e.g., local election shocks in India) ensure the series are not perfectly collinear.

[INSERT IMAGE: lagged_us_vs_india_stress.png]
**Mathematical Interpretation:**
The scatter/correlation diagnostics assess whether stress state probabilities in the US hold predictive association with Indian market stress. The lagged model evaluates $\mathbb{P}(S^{IN}_{t+k} = \text{stress}) \sim f(\mathbb{P}(S^{US}_t = \text{stress}))$. The visual correlation demonstrates a statistically significant but non-deterministic lead-lag relationship, indicating that US volatility shocks frequently precede Indian volatility expansion.

## Market Overlays

[INSERT IMAGE: india_overlay_equity_curves.png]
**Mathematical Interpretation:**
The equity curve overlay integrates the US signal conditionally onto the Indian base strategy. If $\mathbb{P}(S^{US}_t = \text{stress}) > \theta$, Indian exposure $w_t$ is further constrained. The resultant path shows whether absorbing US stress probability acts as an effective leading indicator for Indian drawdowns.

[INSERT IMAGE: india_overlay_exposure.png]
**Mathematical Interpretation:**
The exposure overlay graph visualizes the binary impact of the cross-market signal. It demonstrates mathematically exactly when and where the Indian strategy is forced to de-risk solely due to distress signals originating in the US regime detection ladder.
