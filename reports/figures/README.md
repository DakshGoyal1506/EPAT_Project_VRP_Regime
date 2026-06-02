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
