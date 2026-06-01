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
