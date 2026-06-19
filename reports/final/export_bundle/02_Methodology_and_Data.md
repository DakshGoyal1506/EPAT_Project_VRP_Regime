# 2. Methodology and Data

## Data Sources
The project utilizes daily OHLC price data for the underlying index and daily closing levels for the implied volatility index proxies.
- **US Market:** SPX / SPY and VIX
- **Indian Market:** NIFTY 50 and India VIX

## Realised Variance Estimation
Realised variance is estimated from daily OHLC data. The project relies on the Garman-Klass 22-trading-day annualised realised variance estimator. 

[INSERT IMAGE: rv_estimators_india.png]
**Mathematical Interpretation:**
The Garman-Klass estimator leverages Open ($O_t$), High ($H_t$), Low ($L_t$), and Close ($C_t$) prices to provide a more efficient volatility estimate than close-to-close returns alone. Let $u_t = \ln(H_t/O_t)$, $d_t = \ln(L_t/O_t)$, and $c_t = \ln(C_t/O_t)$. The daily Garman-Klass variance is $\sigma^2_{GK,t} = 0.5(u_t - d_t)^2 - (2\ln 2 - 1)c_t^2$. Over a rolling 22-day window, this is aggregated and annualized ($N=252$): $RV_{t, t+22} = \frac{252}{22} \sum_{i=0}^{21} \sigma^2_{GK, t-i}$. The plot illustrates that OHLC estimators (Parkinson, Garman-Klass) smoothly bound the noisier close-to-close realizations, capturing intraday path variance more effectively.

[INSERT IMAGE: rv_estimators_us.png]
**Mathematical Interpretation:**
Similarly for the US market, the realized variance estimators plot displays the historical path of volatility bounds. The variance compression during calm markets and the massive spikes during crisis windows underscore the non-normality of the underlying return process, motivating the use of dynamic regime models for VRP harvesting.

## Implied Variance and VRP Construction
VIX and India VIX are converted into implied variance proxies by squaring the scaled index:
$IV_t = \left( \frac{VIX_t}{100} \right)^2$

The Variance Risk Premium ($VRP_t$) is mathematically defined as the difference between the risk-neutral expectation of future variance (implied variance) and the physical (realized) variance:
$VRP_t = IV_t - \mathbb{E}_t^\mathbb{P}[RV_{t, t+22}]$

[INSERT IMAGE: india_iv_rv_vrp.png]
**Mathematical Interpretation:**
The comparative overlay of Implied Variance ($IV_t$), Realised Variance ($RV_t$), and the resulting Variance Risk Premium ($VRP_t$) for the Indian market explicitly visualizes the structural premium. During non-crisis periods, $IV_t > RV_t$, yielding a positive $VRP_t$. However, during market crashes, realized variance spikes violently, pushing $RV_t \gg IV_t$ and creating severe, sharp negative VRP realizations. 

[INSERT IMAGE: us_iv_rv_vrp.png]
**Mathematical Interpretation:**
For the US market, the $IV_t$ (derived from VIX) vs $RV_t$ overlay exhibits a similar structural premium. The persistent positive spread represents the compensation investors demand for bearing equity volatility risk. The sudden inversions during periods like the GFC and COVID-19 mathematically represent the left-tail risk inherent in unconditional short-volatility strategies.
