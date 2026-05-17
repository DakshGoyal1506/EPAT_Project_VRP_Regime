# Test Suite

This folder contains the unit and integration-style tests for the EPAT VRP project.

The test suite is organized around the project phases and the most important failure modes:

- data loading and schema validation
- realised variance calculations
- implied variance construction
- calendar alignment and no-forward-fill behavior
- VRP alignment, labels, and no-lookahead registry rules

## Test Philosophy

The tests are designed to protect the project against the most important regression classes:

- lookahead leakage
- duplicate date handling
- incorrect estimator formulas
- wrong market alignment
- invalid registry composition
- accidental inclusion of labels in live features

The suite is intentionally small but specific. A failure should usually point directly to a broken contract in the feature pipeline.

## Test Files

### `test_data_loaders.py`

Covers source loaders and ingestion helpers.

### `test_data_schema.py`

Covers canonical schema expectations and validation rules.

### `test_rv_estimators.py`

Covers realised variance estimator formulas and panel construction.

### `test_implied_variance.py`

Covers IV close-column inference, validation, and annualised implied variance construction.

### `test_calendar_alignment.py`

Covers date alignment, duplicate-date rejection, and calendar mismatch reporting.

### `test_vrp_alignment.py`

Covers IV/RV merging, backward VRP, forward ex-post GK labels, and robustness backward VRP diagnostics.

### `test_no_lookahead.py`

Covers the feature registry firewall and live-feature separation.

## Common Commands

Run the full suite from the repository root:

```bash
pytest
```

Run only the feature-construction tests:

```bash
pytest tests/test_implied_variance.py tests/test_calendar_alignment.py tests/test_rv_estimators.py tests/test_vrp_alignment.py tests/test_no_lookahead.py
```

Run a single file while iterating on a feature module:

```bash
pytest tests/test_vrp_alignment.py -q
```

## Fixtures

Reusable sample data lives in `tests/fixtures/`.

These fixtures are used to keep tests deterministic and small while still exercising the same parsing and alignment logic used by the pipeline.

## Adding New Tests

When adding new feature logic, prefer to add a focused test that checks the public contract rather than only internal helper behavior.

Good tests usually check one of these:

- exact output columns
- exact formulas on small synthetic data
- rejection of invalid input
- preservation of no-lookahead rules
- stable behavior when optional columns are missing
