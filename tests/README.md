# Test Suite

This folder contains the unit and integration-style tests for the EPAT VRP project.

## Test Philosophy

The suite is organized around the project phases and the regression classes that matter most:

- data loading and schema validation
- realised variance calculations
- implied variance construction
- calendar alignment and no-forward-fill behavior
- VRP alignment, labels, and registry rules

The tests are intentionally small but specific. Failures should point to a broken contract in the pipeline, not a vague end-to-end symptom.

## Coverage Map

- `test_data_loaders.py` - source loaders and ingestion helpers.
- `test_data_schema.py` - canonical schema expectations and validation rules.
- `test_rv_estimators.py` - realised variance formulas and panel construction.
- `test_implied_variance.py` - IV close-column inference, validation, and annualised IV construction.
- `test_calendar_alignment.py` - date alignment, duplicate-date rejection, and mismatch reporting.
- `test_vrp_alignment.py` - IV/RV merging, GK VRP, labels, and robustness VRP diagnostics.
- `test_no_lookahead.py` - feature registry firewall and live-feature separation.
- `test_build_features_cli.py` - CLI validation for the Phase 3 22-day VRP contract.

## Common Commands

Run the full suite from the repository root:

```bash
pytest
```

Run the feature-construction slice:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_rv_estimators.py tests/test_vrp_alignment.py tests/test_no_lookahead.py tests/test_build_features_cli.py
```

Run a single file while iterating:

```bash
pytest tests/test_vrp_alignment.py -q
```

## Fixtures

Reusable sample data lives in `tests/fixtures/`. Keep tests deterministic and small while still exercising the same parsing and alignment logic used by the pipeline.

## Adding New Tests

Prefer tests that check the public contract:

- exact output columns
- exact formulas on small synthetic data
- rejection of invalid input
- preservation of no-lookahead rules
- stable behavior when optional columns are missing
