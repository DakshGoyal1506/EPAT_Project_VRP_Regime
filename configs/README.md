# Configs

Configuration files in this folder define the canonical runtime contract for the project.

Use these YAML files as the source of truth for data sources, markets, model settings, backtest parameters, and paper-signal defaults.

Key files:

- `data_sources.yaml` - source priority, ingestion policy, and manual override rules.
- `markets.yaml` - market metadata, calendars, and variance conventions.
- `har_rv.yaml` - HAR-RV forecasting parameters.
- `model_hmm.yaml` - Gaussian HMM regime settings.
- `model_arhmm.yaml` - AR-HMM / Markov autoregression settings.
- `model_markov_autoreg.yaml` - Markov autoregression settings.
- `strategies.yaml` - strategy rules and thresholds.
- `backtest.yaml` - backtest assumptions and accounting rules.
- `ibkr_paper.yaml` - paper-signal adapter defaults and broker guardrails.

Related docs:

- [docs/phase_status.md](../docs/phase_status.md)
- [docs/generated_artifact_policy.md](../docs/generated_artifact_policy.md)
