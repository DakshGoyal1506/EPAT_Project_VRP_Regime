# 1. Abstract and Introduction

## Abstract
The variance risk premium (VRP) is the empirical spread between implied variance and subsequently realised variance. In equity-index markets, implied volatility indices such as VIX and India VIX often stand above subsequent realised volatility, but unconditional short-volatility exposure can suffer severe losses during stress periods.

This project builds a reproducible dual-market research pipeline for the US and India. The US leg uses SPX/SPY and VIX. The India leg uses NIFTY 50 and India VIX. The empirical objective is not to prove live tradability. It is to test whether regime-conditioned exposure improves research-proxy risk-adjusted behaviour relative to unconditional short-volatility harvesting, and whether US and Indian VRP regimes display useful descriptive or predictive relationships.

## Introduction & Executive Summary
Variance risk premium research sits between market microstructure, volatility forecasting, and systematic risk-premia design. Equity-index options often embed a premium for protection demand, crash risk, jump risk, and volatility uncertainty. Short-volatility strategies attempt to harvest that premium, but unconditional exposure is structurally vulnerable to volatility spikes.

The central motivation of this project is to evaluate whether regime conditioning can retain useful premium exposure while reducing exposure during historically adverse states.

The research design has three central components:
1. **VRP measurement:** estimate realised variance from daily OHLC data and compare it with implied variance proxies derived from VIX and India VIX.
2. **Regime decomposition:** classify market states using threshold regimes, Gaussian HMM, Markov autoregression, and MSVOL robustness diagnostics.
3. **Research-proxy strategy evaluation:** compare unconditional short-volatility harvesting with regime-conditioned exposure intentions in a vectorised backtest.
