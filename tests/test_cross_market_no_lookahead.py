from __future__ import annotations

import pandas as pd
import pytest

from vrp.reports.cross_market import (
    CrossMarketInputError,
    CrossMarketLeakageError,
    build_cross_market_no_lookahead_audit,
    validate_no_forbidden_phase11_inputs,
)


def _minimal_config() -> dict:
    return {
        "alignment": {
            "stale_lag_warning_calendar_days": 3,
            "max_lag_calendar_days": 7,
        },
        "forbidden_inputs": {
            "phase11": [
                "reports/tables/phase_11/daily_paper_signal.csv",
                "reports/tables/phase_11/paper_order_intents.csv",
                "reports/tables/phase_11/risk_check_report.csv",
            ]
        },
        "forbidden_keywords": [
            "iBridgePy",
            "paper_order_intents",
            "daily_paper_signal",
            "risk_check_report",
            "broker",
            "paper_signal",
        ],
    }


def test_no_lookahead_audit_passes_with_strict_prior_us_dates() -> None:
    panel = pd.DataFrame(
        {
            "model": ["gaussian_hmm", "gaussian_hmm"],
            "india_date": pd.to_datetime(["2024-01-03", "2024-01-08"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "lag_calendar_days": [1, 3],
        }
    )

    audit = build_cross_market_no_lookahead_audit(
        panel,
        config=_minimal_config(),
        model="gaussian_hmm",
    )

    assert int(audit["n_rows"].iloc[0]) == 2
    assert int(audit["n_same_date_or_future_us_violations"].iloc[0]) == 0
    assert bool(audit["passes_no_lookahead"].iloc[0]) is True


def test_no_lookahead_audit_fails_on_same_date_us_value() -> None:
    panel = pd.DataFrame(
        {
            "model": ["gaussian_hmm"],
            "india_date": pd.to_datetime(["2024-01-02"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02"]),
            "lag_calendar_days": [0],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        build_cross_market_no_lookahead_audit(
            panel,
            config=_minimal_config(),
            model="gaussian_hmm",
        )


def test_no_lookahead_audit_fails_on_future_us_value() -> None:
    panel = pd.DataFrame(
        {
            "model": ["gaussian_hmm"],
            "india_date": pd.to_datetime(["2024-01-02"]),
            "us_lagged_date": pd.to_datetime(["2024-01-03"]),
            "lag_calendar_days": [-1],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        build_cross_market_no_lookahead_audit(
            panel,
            config=_minimal_config(),
            model="gaussian_hmm",
        )


def test_forbidden_phase11_policy_list_is_not_treated_as_configured_input() -> None:
    config = _minimal_config()

    # This should pass because forbidden_inputs is policy, not an active input.
    validate_no_forbidden_phase11_inputs(config)


def test_forbidden_phase11_active_input_is_rejected() -> None:
    config = _minimal_config()
    config["input_files"] = {
        "INDIA": {
            "bad": "reports/tables/phase_11/daily_paper_signal.csv",
        }
    }

    with pytest.raises(CrossMarketInputError):
        validate_no_forbidden_phase11_inputs(config)


def test_no_lookahead_audit_counts_stale_lags_without_same_date_violation() -> None:
    panel = pd.DataFrame(
        {
            "model": ["markov_autoreg", "markov_autoreg", "markov_autoreg"],
            "india_date": pd.to_datetime(["2024-01-03", "2024-01-08", "2024-01-15"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-05"]),
            "lag_calendar_days": [1, 3, 10],
        }
    )

    audit = build_cross_market_no_lookahead_audit(
        panel,
        config=_minimal_config(),
        model="markov_autoreg",
    )

    assert int(audit["n_same_date_or_future_us_violations"].iloc[0]) == 0
    assert int(audit["n_lag_gt_stale_warning_days"].iloc[0]) == 1
    assert int(audit["n_lag_gt_max_lag_days"].iloc[0]) == 1
    assert bool(audit["passes_no_lookahead"].iloc[0]) is True