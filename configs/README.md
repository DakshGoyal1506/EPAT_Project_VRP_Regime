# Configs

This folder contains YAML configuration files used by the research pipeline.

## Rules

1. Configs define source choices, model settings, safety flags, and output paths.
2. Configs should not contain private credentials.
3. Configs should not contain local machine-specific absolute paths.
4. Configs should make source priority and anti-lookahead rules explicit.
5. Broker configs must remain paper/signal-only unless the project is explicitly re-scoped.

## Key Files

| File | Purpose |
|---|---|
| `data_sources.yaml` | Public data source definitions, source priority, local CSV overrides, raw/processed paths |
| `markets.yaml` | US and India market metadata, symbols, timezones, calendar notes |
| `har_rv.yaml` | HAR-RV forecasting configuration |
| `model_hmm.yaml` | Gaussian HMM regime configuration |
| `model_arhmm.yaml` | AR-HMM / Markov autoregression placeholder or legacy config |
| `model_markov_autoreg.yaml` | Markov autoregression regime configuration |
| `strategies.yaml` | Strategy exposure-rule configuration |
| `backtest.yaml` | Backtest input, output, cost, and proxy-return configuration |
| `ibkr_paper.yaml` | Paper-signal broker configuration with live-order blocks |

## Generated Outputs

Configs may point to generated outputs under:

```text
data/processed/
reports/tables/
reports/figures/
data/broker_cache/
```

Those generated files are local-only unless explicitly approved as small final-report artifacts.
