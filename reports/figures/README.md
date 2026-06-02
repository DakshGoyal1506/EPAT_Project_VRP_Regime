# Report Figures

Generated diagnostic and final-report figures are written here.

## Commit Policy

By default, generated figures stay local.

Commit only:

```text
README.md
.gitkeep
selected final-report figures explicitly approved for publication/review
```

Do not commit by default:

```text
*.png
*.svg
*.pdf
*.jpg
*.jpeg
```

## Rules

1. Figures must be reproducible from scripts or `src/vrp/reports/`.
2. Do not manually edit generated figures.
3. Keep exploratory plots local unless selected for the final report.

## Phase 4 Examples

Local Phase 4 figures may include:

```text
har_forecast_us.png
har_forecast_india.png
har_residuals_us.png
har_residuals_india.png
har_vrp_us.png
har_vrp_india.png
```

These are generated diagnostics and stay local unless explicitly approved as final-report artifacts.

## Phase 5 Examples

Local Phase 5 figures may include:

```text
threshold_regimes_us.png
threshold_regimes_india.png
threshold_regime_vrp_boxplots_us.png
threshold_regime_vrp_boxplots_india.png
threshold_component_states_us.png
threshold_component_states_india.png
```

These are generated diagnostics and stay local unless explicitly approved as final-report artifacts.

## Phase 6 Examples

Phase 6 currently does not require committed figures.

If Phase 6 diagnostic figures are generated later, they should live under:

```text
reports/figures/phase_6/
```

They are generated diagnostics and stay local unless explicitly approved as final-report artifacts.

## Phase 7 Examples

If generated later, local Phase 7 figures should live under:

```text
reports/figures/phase_7/
```

They are generated diagnostics and stay local unless explicitly approved as final-report artifacts.

## Phase 9 Examples

If generated later, local Phase 9 figures should live under:

```text
reports/figures/phase_9/
```

They are generated diagnostics and stay local unless explicitly approved as final-report artifacts.

Phase 9 figures, if any, should describe signal availability or exposure distribution only. They must not present backtest performance.

## Phase 8 Examples

If generated later, local Phase 8 figures should live under:

```text
reports/figures/phase_8/
```

They are generated diagnostics and stay local unless explicitly approved as final-report artifacts.
