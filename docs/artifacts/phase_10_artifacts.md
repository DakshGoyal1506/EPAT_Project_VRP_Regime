# Phase 10 Artifacts - Vectorised Research Backtest and Robustness

## Policy

Phase 10 generated outputs are local-only by default.

Commit code, configs, tests, and documentation. Do not commit full backtest panels, generated report tables, generated figures, logs, or broker/runtime files unless explicitly approved as final-report artifacts.

## Artifact Inventory

| Artifact | Local path | Producer command | Commit? | Reason | Expected schema / key columns | Review substitute | Notes |
|---|---|---|---:|---|---|---|---|
| US backtest panel | `data/processed/us_backtest_panel.parquet` | `python scripts/run_backtest.py --market US --strategy all --cost-bps 5 --force` | No | Full generated research panel | `market`, `strategy_name`, `signal_observation_date`, `target_trade_date`, `target_exposure_for_backtest`, `gross_return_proxy`, `cost_proxy`, `net_return_proxy`, `is_backtest_eligible`, `exclusion_reason` | Head/tail/schema preview; final audit JSON | Local-only. |
| India backtest panel | `data/processed/india_backtest_panel.parquet` | `python scripts/run_backtest.py --market INDIA --strategy all --cost-bps 5 --force` | No | Full generated research panel | Same as US | Head/tail/schema preview; final audit JSON | Local-only. |
| US metadata sidecar | `data/processed/us_backtest_panel_metadata.json` | Same as US backtest panel | No by default | Generated run metadata | `phase`, `payoff_label`, `label_role`, `outcome_alignment`, `cost_bps`, `overlapping_labels`, no-lookahead counts | JSON preview | May be quoted in report, but file stays local by default. |
| India metadata sidecar | `data/processed/india_backtest_panel_metadata.json` | Same as India backtest panel | No by default | Generated run metadata | Same as US metadata | JSON preview | Local-only by default. |
| Input audit CSV | `reports/tables/phase_10/phase10_input_audit.csv` | `python scripts/audit_phase10_inputs.py --market ALL` | No by default | Generated input validation output | `market`, `component`, `path`, `validation_status`, `details` | Terminal summary or CSV preview | Small but generated. |
| Input audit JSON | `reports/tables/phase_10/phase10_input_audit.json` | Same as input audit CSV | No by default | Generated input validation metadata | `has_errors`, `n_errors`, `n_warnings`, `issues` | JSON preview | Critical audit artifact. |
| Backtest summary | `reports/tables/phase_10/backtest_summary.csv` | `python scripts/generate_backtest_diagnostics.py --market ALL` | No by default | Generated strategy metric summary | `market`, `strategy_name`, `n_obs`, `n_eligible`, `total_return_proxy`, `sharpe`, `max_drawdown`, `return_to_drawdown` | CSV preview | Candidate final-report table if explicitly approved. |
| Common-start summary | `reports/tables/phase_10/backtest_common_start_summary.csv` | Same diagnostics command | No by default | Fair comparison across strategies with different start dates | `common_start_date`, metrics columns | CSV preview | Candidate final-report table. |
| Tail summary | `reports/tables/phase_10/backtest_tail_summary.csv` | Same diagnostics command | No by default | Strategy-level tail behavior | `p01`, `p05`, `p50`, `p95`, `p99`, `worst_return`, `cte_95` | CSV preview | Useful for report risk discussion. |
| Strategy-year table | `reports/tables/phase_10/backtest_by_strategy_year.csv` | Same diagnostics command | No | Large generated diagnostic table | `market`, `strategy_name`, `year`, metrics columns | Selected year preview | Local-only. |
| Crisis-window performance | `reports/tables/phase_10/crisis_window_performance.csv` | Same diagnostics command | No by default | Pre-specified stress-window performance | `market`, `subperiod`, `start_date`, `end_date`, `strategy_name`, metrics columns | CSV preview | Candidate final-report excerpt. |
| Availability summary | `reports/tables/phase_10/backtest_availability_summary.csv` | Same diagnostics command | No | Generated availability diagnostic | `market`, `strategy_name`, `n_obs`, `n_eligible`, `availability_rate`, exclusion counts | CSV preview | Local-only by default. |
| No-lookahead audit | `reports/tables/phase_10/backtest_no_lookahead_audit.csv` | Same diagnostics command | No by default | Timing audit | no-lookahead violation counts, `passes_no_lookahead_audit` | CSV preview | Critical review substitute. |
| Backtest metadata | `reports/tables/phase_10/backtest_metadata.json` | Same diagnostics command | No by default | Generated report metadata | `phase`, `markets`, `payoff_label`, `limitations`, `common_start_dates`, `no_lookahead_audit_passed` | JSON preview | Critical review substitute. |
| Cost sensitivity | `reports/tables/phase_10/robustness_cost_sensitivity.csv` | `python scripts/run_robustness.py --market ALL --test costs --force` | No by default | Generated robustness table | `cost_bps`, `market`, `strategy_name`, metrics columns | CSV preview | Candidate report excerpt. |
| Subperiod robustness | `reports/tables/phase_10/robustness_subperiods.csv` | `python scripts/run_robustness.py --market ALL --test subperiod --force` | No by default | Generated robustness table | `subperiod`, `subperiod_start`, `subperiod_end`, metrics columns | CSV preview | Candidate report excerpt. |
| Weekly rebalance skip report | `reports/tables/phase_10/robustness_weekly_rebalance_skipped.json` | `python scripts/run_robustness.py --market ALL --test weekly --force` | No by default | Skip-safe documentation | `status`, `reason` | JSON preview | Documents intentional skip. |
| Tradable proxy detection | `reports/tables/phase_10/tradable_proxy_detection.json` | `python scripts/run_robustness.py --market ALL --test tradable_proxy --force` | No by default | Detects existing local tradable proxy files only | `status`, `reason`, `n_candidates` | JSON preview | No downloads. |
| Robustness metadata | `reports/tables/phase_10/robustness_metadata.json` | `python scripts/run_robustness.py --market ALL --test all --force` | No by default | Robustness run summary | `tests`, `rules` | JSON preview | Critical audit substitute. |
| Final audit JSON | `reports/tables/phase_10/phase10_final_audit.json` | `python scripts/audit_phase10_final.py --market ALL` | No by default | Final artifact audit | `status`, `n_errors`, `n_warnings`, `issues` | JSON preview | Critical review substitute. |
| Equity curves | `reports/figures/phase_10/equity_curves_us.png`, `reports/figures/phase_10/equity_curves_india.png` | `python scripts/generate_backtest_diagnostics.py --market ALL` | No by default | Generated diagnostic figures | N/A | Screenshot if needed | Additive research proxy curves only. |
| Common-start equity curves | `reports/figures/phase_10/equity_curves_common_start_us.png`, `reports/figures/phase_10/equity_curves_common_start_india.png` | Same diagnostics command | No by default | Fair visual comparison | N/A | Screenshot if needed | Better for report comparison. |
| Drawdowns | `reports/figures/phase_10/drawdowns_us.png`, `reports/figures/phase_10/drawdowns_india.png` | Same diagnostics command | No by default | Generated drawdown diagnostics | N/A | Screenshot if needed | Proxy drawdowns, not account drawdowns. |
| Return distributions | `reports/figures/phase_10/return_distribution_us.png`, `reports/figures/phase_10/return_distribution_india.png` | Same diagnostics command | No by default | Generated return distribution diagnostics | N/A | Screenshot if needed | Pooled distribution; use tail table for strategy-level tails. |

## Sensitivity and Reproducibility Notes

- Phase 10 outputs depend on local Phase 9 signal panels.
- Backtest panels are generated from fixed signals; Phase 10 must not mutate Phase 9 signal logic.
- The payoff label is realised outcome only.
- No generated artifact should contain broker credentials or live trading state.
- Full generated panels stay local.
- Final-report-selected small summaries or figures require explicit approval before commit.
