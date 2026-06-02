# Known Limitations

This document lists current limitations, non-goals, and assumptions. It should be updated when a limitation is fixed or accepted as final.

## Data Source Limitations

1. Public data vendors can change formats, ticker availability, history depth, or download behaviour.
2. Yahoo Finance data via `yfinance` is convenient but not official exchange data.
3. NSE scripted access may be blocked or rate-limited; manual CSV override is intentionally supported.
4. CBOE/FRED/Yahoo VIX series can differ slightly because of source methodology, close timing, and revisions.
5. Official/manual source downloads must be documented locally if used.
6. The project does not commit raw data, so reviewers must regenerate or inspect summaries.

## Calendar and Alignment Limitations

1. US and India have different trading calendars.
2. Cross-market alignment must be explicit and phase-specific.
3. Early phases must not silently merge US and India dates.
4. Holiday gaps are expected and should not be automatically forward-filled.
5. Date-level daily storage is used; intraday timezone mechanics are out of scope for core public-data phases.

## Realised Variance Limitations

1. Daily OHLC estimators are proxies for realised variance.
2. Range-based estimators can be sensitive to bad high/low data.
3. Close-to-close volatility ignores intraday range.
4. Daily OHLC does not capture true intraday realised variance.
5. The selected estimator must remain consistent when comparing against annualized implied variance.

## VRP Construction Limitations

1. VIX and India VIX are implied-volatility index proxies, not direct variance swap quotes.
2. `VIX^2` and `India VIX^2` are model-free proxy approximations for annualized implied variance.
3. Forward ex-post realised variance labels are outcomes, not tradable signals.
4. Any prospective VRP signal must use only information available at the signal timestamp.

## Forecasting Limitations

1. HAR-RV forecasts depend on the realised-variance estimator chosen upstream.
2. Forecasts are point estimates, not executable trade guarantees.
3. Expanding/rolling windows must be audited for no-lookahead leakage.
4. GPU acceleration is optional and not required for reproducibility.
5. HAC coefficient inference can be disabled for faster production forecast runs.

## Regime Model Limitations

1. Gaussian HMM emissions do not directly model autoregression in observed VRP/RV series.
2. State labels are economic interpretations, not observed truth.
3. Full-sample smoothed probabilities are diagnostic only.
4. Backtests must use filtered probabilities available at time `t`.
5. State-count selection can be sample-sensitive.
6. Crisis-window interpretation must not be used as a hidden training label unless explicitly documented.
7. Phase 5 threshold regimes are deterministic baseline labels, not observed truth.
8. Phase 5 calm states may be sparse because calm requires several conservative conditions to pass simultaneously.
9. Phase 5 crisis-window diagnostics are reporting-only and must not be used for threshold calibration.
10. Markov autoregression improves on Gaussian HMM by directly modelling observed-series autocorrelation, but fitting can be numerically sensitive and may require train-only scaling or winsorization.
11. Phase 7 MAR stress states are economic interpretations based on volatility, IV/RV, target variance, and return behavior; they need not imply lower prospective HAR-based VRP mean in every market/sample.
12. HMM/threshold agreement with MAR is diagnostic only and must not be used as a model-selection objective.
13. Dynamax/JAX AR-HMM support is optional and must remain stub-only unless explicitly re-scoped.
14. Phase 8 MSVOL is a Python-only Markov-switching variance robustness proxy, not true MSGARCH.
15. True MSGARCH remains optional/future because it requires the R `MSGARCH` package.
16. MSVOL models return-volatility regimes only and is diagnostic-only.
17. MSVOL outputs must not be used for Phase 9 strategy construction, backtesting, VaR/ES, or cross-market analysis.
18. MSVOL smoothed probabilities, if present, are diagnostic-only; only filtered probabilities may be considered time-safe.

## Backtest Limitations

1. Backtest returns are research-layer proxy returns unless explicitly converted to executable instruments.
2. Forward VRP labels are realised outcomes, not available at trade time.
3. Transaction costs are approximations unless tied to instrument-level execution data.
4. Overlapping horizons can inflate dependence in returns and metrics.
5. Results must be interpreted as empirical evidence, not live-trading proof.
6. Phase 10 cumulative curves are additive research proxy sums, not executable account equity curves.
7. Phase 10 does not define initial capital, margin, or percentage return on invested capital.

## Broker / Paper-Signal Limitations

1. iBridgePy / IBKR is optional and not required for core research reproducibility.
2. Broker integration is paper/signal-only by default.
3. Live orders are disabled by config and policy.
4. Broker account identifiers must remain local.
5. Instrument availability and permissions must be verified outside the research pipeline.
6. The project does not infer live option contract sizing from research proxy backtests.

## Documentation Limitations

1. Phase status can drift if `docs/phase_status.md` is not updated after changes.
2. Artifact inventory is a governance document, not an automated manifest.
3. Local generated artifacts may exist outside Git and must be reviewed through summaries or command output.
