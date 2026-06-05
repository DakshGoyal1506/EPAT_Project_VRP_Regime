from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.reports.cross_market import (
    CrossMarketLeakageError,
    align_us_india_predictive_panel,
    assert_no_same_date_us_leakage,
    build_alignment_audit,
    previous_us_trading_date_for_india,
)


def _minimal_config(max_lag_calendar_days: int | None = 7) -> dict:
    return {
        "alignment": {
            "method": "asof_backward_strict",
            "allow_exact_matches": False,
            "require_us_lagged_date_lt_india_date": True,
            "max_lag_calendar_days": max_lag_calendar_days,
            "stale_lag_warning_calendar_days": 3,
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


def test_previous_us_trading_date_is_strictly_before_india_date() -> None:
    india_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"])
    us_dates = pd.to_datetime(["2024-01-02", "2024-01-05"])

    aligned = previous_us_trading_date_for_india(india_dates, us_dates)

    assert pd.isna(aligned.loc[0, "us_lagged_date"])
    assert aligned.loc[1, "us_lagged_date"] == pd.Timestamp("2024-01-02")
    assert aligned.loc[2, "us_lagged_date"] == pd.Timestamp("2024-01-05")

    matched = aligned["us_lagged_date"].notna()
    assert (aligned.loc[matched, "us_lagged_date"] < aligned.loc[matched, "india_date"]).all()

    assert_no_same_date_us_leakage(aligned)


def test_previous_us_trading_date_handles_mixed_datetime_units() -> None:
    india_dates = pd.Series(
        np.array(["2024-01-02", "2024-01-03", "2024-01-08"], dtype="datetime64[ms]")
    )
    us_dates = pd.Series(
        np.array(["2024-01-02", "2024-01-05"], dtype="datetime64[us]")
    )

    aligned = previous_us_trading_date_for_india(india_dates, us_dates)

    assert str(aligned["india_date"].dtype) == "datetime64[ns]"
    assert str(aligned["us_lagged_date"].dtype) == "datetime64[ns]"
    assert pd.isna(aligned.loc[0, "us_lagged_date"])
    assert aligned.loc[1, "us_lagged_date"] == pd.Timestamp("2024-01-02")
    assert aligned.loc[2, "us_lagged_date"] == pd.Timestamp("2024-01-05")

    assert_no_same_date_us_leakage(aligned)


def test_assert_no_same_date_us_leakage_rejects_same_date() -> None:
    panel = pd.DataFrame(
        {
            "india_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "lag_calendar_days": [0, 1],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        assert_no_same_date_us_leakage(panel)


def test_assert_no_same_date_us_leakage_rejects_future_us_date() -> None:
    panel = pd.DataFrame(
        {
            "india_date": pd.to_datetime(["2024-01-02"]),
            "us_lagged_date": pd.to_datetime(["2024-01-03"]),
            "lag_calendar_days": [-1],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        assert_no_same_date_us_leakage(panel)


def test_align_us_india_predictive_panel_blocks_exact_matches() -> None:
    us_df = pd.DataFrame(
        {
            "us_date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "us_vrp_har_gk": [1.0, 2.0],
            "us_stress_prob": [0.25, 0.75],
        }
    )
    india_df = pd.DataFrame(
        {
            "india_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-08"]),
            "india_vrp_har_gk": [10.0, 11.0, 12.0],
            "india_stress_prob": [0.1, 0.2, 0.3],
        }
    )

    panel = align_us_india_predictive_panel(
        us_df=us_df,
        india_df=india_df,
        config=_minimal_config(max_lag_calendar_days=7),
    )

    assert pd.isna(panel.loc[0, "us_lagged_date"])
    assert panel.loc[1, "us_lagged_date"] == pd.Timestamp("2024-01-02")
    assert panel.loc[2, "us_lagged_date"] == pd.Timestamp("2024-01-05")

    matched = panel["us_lagged_date"].notna()
    assert (panel.loc[matched, "us_lagged_date"] < panel.loc[matched, "india_date"]).all()


def test_align_us_india_predictive_panel_fails_on_stale_lag_beyond_config() -> None:
    us_df = pd.DataFrame(
        {
            "us_date": pd.to_datetime(["2024-01-01"]),
            "us_vrp_har_gk": [1.0],
            "us_stress_prob": [0.25],
        }
    )
    india_df = pd.DataFrame(
        {
            "india_date": pd.to_datetime(["2024-01-15"]),
            "india_vrp_har_gk": [10.0],
            "india_stress_prob": [0.1],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        align_us_india_predictive_panel(
            us_df=us_df,
            india_df=india_df,
            config=_minimal_config(max_lag_calendar_days=7),
        )


def test_build_alignment_audit_counts_missing_and_stale_lags() -> None:
    panel = pd.DataFrame(
        {
            "model": ["test"] * 4,
            "india_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-15"]
            ),
            "us_lagged_date": pd.to_datetime(
                [pd.NaT, "2024-01-02", "2024-01-05", "2024-01-05"]
            ),
            "lag_calendar_days": [np.nan, 1, 3, 10],
        }
    )

    audit = build_alignment_audit(panel, model="test")

    assert int(audit["n_india_dates"].iloc[0]) == 4
    assert int(audit["n_matched_us_lagged_dates"].iloc[0]) == 3
    assert int(audit["n_missing_us_lagged_dates"].iloc[0]) == 1
    assert int(audit["n_lag_gt_3_calendar_days"].iloc[0]) == 1
    assert int(audit["n_lag_gt_7_calendar_days"].iloc[0]) == 1
    assert int(audit["n_same_date_violations"].iloc[0]) == 0


def test_build_alignment_audit_fails_on_same_date_violation() -> None:
    panel = pd.DataFrame(
        {
            "model": ["test"],
            "india_date": pd.to_datetime(["2024-01-02"]),
            "us_lagged_date": pd.to_datetime(["2024-01-02"]),
            "lag_calendar_days": [0],
        }
    )

    with pytest.raises(CrossMarketLeakageError):
        build_alignment_audit(panel, model="test")