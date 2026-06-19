# 3. Predictive Volatility Modeling

## HAR-RV Framework
The HAR-RV (Heterogeneous Autoregressive model of Realised Volatility) estimates prospective realised variance using lagged realised-variance features. Its role is to provide a model-based expected realised variance $\mathbb{E}_t[RV_{t, t+22}]$ input for prospective VRP construction under point-in-time constraints.

[INSERT IMAGE: har_forecast_india.png]
**Mathematical Interpretation:**
The HAR-RV model constructs a prospective realized variance forecast $RV_{t, t+22}$ using lagged daily, weekly, and monthly realized variance components:
$RV_{t, t+22} = \beta_0 + \beta_d RV^{(d)}_t + \beta_w RV^{(w)}_t + \beta_m RV^{(m)}_t + \epsilon_{t+22}$
The forecast plot demonstrates that while the model captures the low-frequency volatility persistence and mean-reversion characteristic of the market, it structurally lags behind sudden, explosive volatility shocks.

[INSERT IMAGE: har_forecast_us.png]
**Mathematical Interpretation:**
For the US market, the HAR-RV forecast effectively models the long-memory properties of volatility. The cascading beta weights ($\beta_d > \beta_w > \beta_m$) ensure that recent shocks decay properly while maintaining a long-term anchor. However, it functions as a point-in-time forecast and struggles with jump-diffusion elements inherent in real market variance.

[INSERT IMAGE: har_residuals_india.png]
**Mathematical Interpretation:**
The residuals (forecast errors) $\epsilon_t = RV_{t, t+22} - \widehat{RV}_{t, t+22}$ for the Indian market exhibit pronounced heteroskedasticity and fat tails during volatility spikes. A perfectly specified model would yield white noise residuals, but the clustering of large errors indicates non-linear regime shifts that linear autoregressive structures fail to anticipate.

[INSERT IMAGE: har_residuals_us.png]
**Mathematical Interpretation:**
Similarly, the US market HAR-RV residuals highlight structural breaks where actual variance significantly exceeds forecasted variance. These concentrated error clusters are exactly the regimes where unconditional short-volatility strategies suffer maximum drawdowns, motivating the application of regime-detection models.

[INSERT IMAGE: har_vrp_india.png]
**Mathematical Interpretation:**
The HAR-based prospective VRP for India utilizes the HAR forecast as the expectation component: $VRP^{HAR}_t = IV_t - \widehat{RV}^{HAR}_{t, t+22}$. This isolates a tradable proxy of the premium without lookahead bias.

[INSERT IMAGE: har_vrp_us.png]
**Mathematical Interpretation:**
The HAR-based prospective VRP for the US visualizes the available risk premium at time $t$. Unlike ex-post VRP which uses future information, this signal is point-in-time compliant and forms the bedrock feature for the subsequent regime classification algorithms.
