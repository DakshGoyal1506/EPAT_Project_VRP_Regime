# tests/test_calendar_alignment.py

from __future__ import annotations

import pandas as pd
import pytest

from vrp.features.calendars import (
    align_market_dates,
    build_calendar_mismatch_table,
    report_calendar_mismatches,
)


def make_iv_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "market": "US",
            "iv_symbol": "VIX",
            "iv_close": [10.0, 20.0, 30.0],
            "iv_ann": [0.01, 0.04, 0.09],
        }
    )


def make_rv_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                ]
            ),
            "market": "US",
            "symbol": "US_UNDERLYING",
            "rv_gk_daily": [0.001, 0.002, 0.003],
            "rv_gk_22d_ann": [0.10, 0.20, 0.30],
        }
    )


def test_align_market_dates_inner_join_only() -> None:
    iv = make_iv_panel()
    rv = make_rv_panel()

    aligned = align_market_dates(iv, rv)

    assert aligned["date"].tolist() == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]

    assert len(aligned) == 2
    assert aligned["iv_close"].tolist() == [20.0, 30.0]
    assert aligned["rv_gk_daily"].tolist() == [0.001, 0.002]


def test_align_market_dates_does_not_forward_fill() -> None:
    iv = make_iv_panel()
    rv = make_rv_panel()

    aligned = align_market_dates(iv, rv)

    assert pd.Timestamp("2020-01-01") not in set(aligned["date"])
    assert pd.Timestamp("2020-01-04") not in set(aligned["date"])


def test_report_calendar_mismatches_counts_dates() -> None:
    iv = make_iv_panel()
    rv = make_rv_panel()

    row = report_calendar_mismatches(iv, rv, market="US")

    assert row["market"] == "US"
    assert row["iv_rows"] == 3
    assert row["rv_rows"] == 3
    assert row["common_dates"] == 2
    assert row["iv_only_dates"] == 1
    assert row["rv_only_dates"] == 1
    assert row["first_iv_only_date"] == "2020-01-01"
    assert row["first_rv_only_date"] == "2020-01-04"


def test_build_calendar_mismatch_table_has_stable_columns() -> None:
    iv = make_iv_panel()
    rv = make_rv_panel()

    row = report_calendar_mismatches(iv, rv, market="US")
    table = build_calendar_mismatch_table([row])

    assert list(table.columns) == [
        "market",
        "iv_start",
        "iv_end",
        "rv_start",
        "rv_end",
        "iv_rows",
        "rv_rows",
        "common_dates",
        "iv_only_dates",
        "rv_only_dates",
        "first_iv_only_date",
        "first_rv_only_date",
    ]


def test_align_market_dates_rejects_duplicate_iv_dates() -> None:
    iv = pd.concat([make_iv_panel(), make_iv_panel().iloc[[0]]], ignore_index=True)
    rv = make_rv_panel()

    with pytest.raises(ValueError, match="duplicated"):
        align_market_dates(iv, rv)


def test_align_market_dates_rejects_duplicate_rv_dates() -> None:
    iv = make_iv_panel()
    rv = pd.concat([make_rv_panel(), make_rv_panel().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicated"):
        align_market_dates(iv, rv)


def test_align_market_dates_rejects_market_mismatch() -> None:
    iv = make_iv_panel()
    rv = make_rv_panel()
    rv["market"] = "INDIA"

    with pytest.raises(ValueError, match="market mismatch"):
        align_market_dates(iv, rv)