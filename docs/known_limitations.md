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

## Cross-Market Analysis Limitations

1. Phase 13 uses daily close-level data and does not model intraday timezone mechanics.
2. Predictive India diagnostics use the latest US observation strictly before the India date.
3. Same-date US/India comparisons are descriptive only.
4. US and India holiday gaps may create stale lagged US observations.
5. Granger-style diagnostics are lead-lag statistical diagnostics, not causal proof.
6. The Phase 13 overlay is analysis-only and not part of the Phase 9 strategy universe.
7. Overlay results do not constitute live-trading or execution evidence.

## Broker / Paper-Signal Limitations

1. iBridgePy / IBKR is optional and not required for core research reproducibility.
2. Broker integration is paper/signal-only by default.
3. Live orders are disabled by config and policy.
4. Broker account identifiers must remain local.
5. Instrument availability and permissions must be verified outside the research pipeline.
6. The project does not infer live option contract sizing from research proxy backtests.

## Phase 12 Limitation

1. Phase 12 was intentionally skipped.
2. Phase 12 remains future optional work only.
3. The IBKR paper execution adapter was not implemented in the current submission scope.
4. Do not describe Phase 12 as not started, partially implemented, implemented, or validated.

Required wording:

```text
Phase 12 = skipped / future optional — IBKR paper execution adapter intentionally left out of current submission scope.
```

## Documentation Limitations

1. Phase status can drift if `docs/phase_status.md` is not updated after changes.
2. Artifact inventory is a governance document, not an automated manifest.
3. Local generated artifacts may exist outside Git and must be reviewed through summaries or command output.

## Final Report Limitations

1. Final report numeric findings must be inserted only after inspecting local result tables.

2. Every major report claim must map to `reports/final/result_claims_audit.md`.

3. Every final report table must map to `reports/final/table_inventory.md`.

4. Every final report figure must map to `reports/final/figure_inventory.md`.

5. `reports/final/final_report.md` is the source of truth.

6. `reports/final/final_report.pdf` is an export deliverable.

7. The PDF must not contain unaudited numeric claims, broker-sensitive artifacts, or full generated panels.
