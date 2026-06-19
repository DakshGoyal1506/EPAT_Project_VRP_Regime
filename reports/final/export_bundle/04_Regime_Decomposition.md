# 4. Regime Decomposition

The regime modelling ladder addresses the state classification of the market using progressively sophisticated models:
- **Threshold Regimes:** Provide a simple, deterministic baseline.
- **Gaussian HMM:** Estimates latent regimes from observed features using filtered probabilities.
- **Markov Autoregression (MAR):** Extends the regime ladder by allowing state-dependent autoregressive dynamics in the observed series, addressing autocorrelation constraints in standard HMMs.

## Threshold Regimes

[INSERT IMAGE: threshold_regimes_india.png]
**Mathematical Interpretation:**
The threshold approach deterministically classifies the market state $S_t \in \{\text{calm}, \text{transition}, \text{stress}\}$ based on explicit boundaries of the prospective VRP and underlying variance levels. The plot overlays these states over time, indicating periods where variance risk compensation is sufficient versus periods where standard deviations aggressively breach historical norms.

[INSERT IMAGE: threshold_regimes_us.png]
**Mathematical Interpretation:**
For the US market, the threshold regimes provide a hard-filter historical baseline. It isolates explicit stress periods (e.g., GFC, Volmageddon, COVID-19) strictly from historical feature distributions, serving as the benchmark against which probabilistic HMMs will be evaluated.

[INSERT IMAGE: threshold_component_states_india.png]
**Mathematical Interpretation:**
This plot disaggregates the deterministic state vectors. It visually verifies that the strict logical bounds applied to $IV_t$ and $RV_t$ produce the required hard constraints for identifying periods of insufficient premium carry.

[INSERT IMAGE: threshold_component_states_us.png]
**Mathematical Interpretation:**
The US component states similarly validate the boolean masking. The isolation of 'stress' explicitly truncates exposure during the most severe historical market events.

## Regime Conditioning and Premium Distributions

[INSERT IMAGE: threshold_regime_vrp_boxplots_india.png]
**Mathematical Interpretation:**
The boxplots visualize the empirical distribution of the Variance Risk Premium ($VRP_t = IV_{t, t+22} - \mathbb{E}_t[RV_{t, t+22}]$) conditioned on the deterministic threshold regimes. We observe that during 'stress' regimes, the VRP distribution exhibits significantly higher dispersion and a thicker right tail compared to 'calm' and 'transition' regimes. Mathematically, this indicates that while volatility risk premia expand during turbulent periods to compensate for jump and crash risks, the variance of the premium itself $\text{Var}(VRP_t | S_t = \text{stress})$ also significantly increases, reducing risk-adjusted predictability.

[INSERT IMAGE: threshold_regime_vrp_boxplots_us.png]
**Mathematical Interpretation:**
The US market boxplots confirm the same phenomenon. The calm regime demonstrates tight clustering around a positive median premium, while the stress regime exhibits immense variance and structural left-skewed outliers, perfectly illustrating why short-volatility exposure must be regime-conditioned to optimize the Sortino ratio.
