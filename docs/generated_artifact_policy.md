# Generated Artifact Policy

## Commit Policy

Commit source code, configuration, documentation, tests, and lightweight README files.

Do not commit generated data panels, model binaries, logs, or broker cache artifacts.

## Local-Only Rules

The following stay local unless explicitly approved for a release artifact:

```text
data/raw/*
data/interim/*
data/processed/*
reports/figures/*
reports/tables/*
logs/*
broker_cache/*
model binaries
notebook output runs
```

## Selected Small-Summary Exception

Small, stable summary CSV or JSON artifacts may be committed if they are clearly useful for review and do not expose broker-sensitive state or large generated panels.
