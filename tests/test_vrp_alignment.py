# tests/test_vrp_alignment.py

from __future__ import annotations

import pandas as pd
import pytest

from vrp.features.vrp import (
    build_vrp_panel,
    compute_backward_vrp,
    compute_backward_vrp_robustness,
    compute_forward_expost_vrp,
    merge_iv_rv,
)


def make_iv_panel(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    return pd.DataFrame(
        {
            "date": dates,
            "market": "US",
            "iv_symbol": "VIX",
            "iv_close": 20.0,
            "iv_ann": 0.04,
        }
    )


def make_rv_panel(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Unique daily values make forward-label off-by-one errors obvious.
    rv_daily = [(i + 1) / 252.0 for i in range(n)]

    return pd.DataFrame(
        {
            "date": dates,
            "market": "US",
            "symbol": "US_UNDERLYING",
            "log_return": 0.0,
            "simple_return": 0.0,
            "rv_gk_daily": rv_daily,
            "rv_gk_22d_ann": [0.10 + i for i in range(n)],
            "rv_cc_22d_ann": [0.20 + i for i in range(n)],
            "rv_parkinson_22d_ann": [0.30 + i for i in range(n)],
            "rv_rs_22d_ann": [0.40 + i for i in range(n)],
            "rv_yz_22d_ann": [0.50 + i for i in range(n)],
        }
    )


def test_merge_iv_rv_inner_join_only() -> None:
    iv = make_iv_panel(n=5)
    rv = make_rv_panel(n=5)

    iv = iv.iloc[:4].copy()
    rv = rv.iloc[1:].copy()

    merged = merge_iv_rv(iv, rv, market="US")

    assert merged["date"].tolist() == list(pd.date_range("2020-01-02", periods=3, freq="B"))


def test_backward_vrp_uses_lagged_rv_not_same_day_rv() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="B"),
            "iv_ann": [100.0, 100.0, 100.0, 100.0],
            "rv_gk_22d_ann": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = compute_backward_vrp(panel, rv_col="rv_gk_22d_ann")

    assert pd.isna(out.loc[0, "rv_gk_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_gk"])

    assert out.loc[1, "rv_gk_22d_ann_lag1"] == pytest.approx(1.0)
    assert out.loc[1, "vrp_backward_gk"] == pytest.approx(99.0)

    assert out.loc[2, "rv_gk_22d_ann_lag1"] == pytest.approx(2.0)
    assert out.loc[2, "vrp_backward_gk"] == pytest.approx(98.0)

    assert out.loc[3, "rv_gk_22d_ann_lag1"] == pytest.approx(3.0)
    assert out.loc[3, "vrp_backward_gk"] == pytest.approx(97.0)


def test_backward_vrp_robustness_uses_lagged_rv_not_same_day() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="B"),
            "iv_ann": [100.0] * 5,
            "rv_cc_22d_ann": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rv_parkinson_22d_ann": [10.0, 20.0, 30.0, 40.0, 50.0],
            "rv_rs_22d_ann": [100.0, 200.0, 300.0, 400.0, 500.0],
            "rv_yz_22d_ann": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )

    out = compute_backward_vrp_robustness(panel)

    assert pd.isna(out.loc[0, "rv_cc_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_cc"])
    assert pd.isna(out.loc[0, "rv_parkinson_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_parkinson"])
    assert pd.isna(out.loc[0, "rv_rs_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_rs"])
    assert pd.isna(out.loc[0, "rv_yz_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_yz"])

    assert out.loc[1, "rv_cc_22d_ann_lag1"] == pytest.approx(1.0)
    assert out.loc[1, "vrp_backward_cc"] == pytest.approx(99.0)
    assert out.loc[1, "rv_parkinson_22d_ann_lag1"] == pytest.approx(10.0)
    assert out.loc[1, "vrp_backward_parkinson"] == pytest.approx(90.0)
    assert out.loc[1, "rv_rs_22d_ann_lag1"] == pytest.approx(100.0)
    assert out.loc[1, "vrp_backward_rs"] == pytest.approx(0.0)
    assert out.loc[1, "rv_yz_22d_ann_lag1"] == pytest.approx(1000.0)
    assert out.loc[1, "vrp_backward_yz"] == pytest.approx(-900.0)


def test_forward_expost_vrp_uses_t_plus_1_through_horizon() -> None:
    horizon = 3
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="B"),
            "iv_ann": [100.0] * 8,
            "rv_gk_daily": [
                1.0 / 252.0,
                2.0 / 252.0,
                3.0 / 252.0,
                4.0 / 252.0,
                5.0 / 252.0,
                6.0 / 252.0,
                7.0 / 252.0,
                8.0 / 252.0,
            ],
        }
    )

    out = compute_forward_expost_vrp(
        panel,
        rv_daily_col="rv_gk_daily",
        horizon=horizon,
        annualization_periods=252,
    )

    # At t=0, use days 1,2,3 => values 2,3,4 after annualisation scale.
    expected_forward_rv_t0 = (2.0 + 3.0 + 4.0) / 3.0
    assert out.loc[0, "rv_gk_22d_forward_ann_label"] == pytest.approx(
        expected_forward_rv_t0
    )
    assert out.loc[0, "vrp_forward_expost_gk_label"] == pytest.approx(
        100.0 - expected_forward_rv_t0
    )

    # At t=1, use days 2,3,4 => values 3,4,5.
    expected_forward_rv_t1 = (3.0 + 4.0 + 5.0) / 3.0
    assert out.loc[1, "rv_gk_22d_forward_ann_label"] == pytest.approx(
        expected_forward_rv_t1
    )

    # Last horizon rows cannot have full future label.
    assert out["rv_gk_22d_forward_ann_label"].iloc[-3:].isna().all()
    assert out["vrp_forward_expost_gk_label"].iloc[-3:].isna().all()


def test_forward_expost_columns_are_label_named() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=8, freq="B"),
            "iv_ann": [100.0] * 8,
            "rv_gk_daily": [1.0 / 252.0] * 8,
        }
    )

    out = compute_forward_expost_vrp(
        panel,
        rv_daily_col="rv_gk_daily",
        horizon=3,
        annualization_periods=252,
    )

    assert "rv_gk_22d_forward_ann_label" in out.columns
    assert "vrp_forward_expost_gk_label" in out.columns

    assert "forward" in "rv_gk_22d_forward_ann_label"
    assert "label" in "rv_gk_22d_forward_ann_label"
    assert "forward" in "vrp_forward_expost_gk_label"
    assert "expost" in "vrp_forward_expost_gk_label"
    assert "label" in "vrp_forward_expost_gk_label"


def test_build_vrp_panel_required_columns_and_feature_allowed() -> None:
    iv = make_iv_panel(n=30)
    rv = make_rv_panel(n=30)

    out = build_vrp_panel(
        iv_df=iv,
        rv_df=rv,
        market="US",
        horizon=3,
        annualization_periods=252,
    )

    required = [
        "date",
        "market",
        "underlying_symbol",
        "iv_symbol",
        "iv_close",
        "iv_ann",
        "rv_gk_daily",
        "rv_gk_22d_ann",
        "rv_gk_22d_ann_lag1",
        "vrp_backward_gk",
        "vrp_backward_gk_positive",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "feature_allowed",
    ]

    for col in required:
        assert col in out.columns

    assert out["market"].eq("US").all()
    assert out["underlying_symbol"].eq("US_UNDERLYING").all()

    assert pd.isna(out.loc[0, "rv_gk_22d_ann_lag1"])
    assert pd.isna(out.loc[0, "vrp_backward_gk"])

    assert out["feature_allowed"].dtype == bool
    assert not bool(out.loc[0, "feature_allowed"])
    assert bool(out.loc[1, "feature_allowed"])


def test_build_vrp_panel_preserves_robustness_columns_when_available() -> None:
    iv = make_iv_panel(n=30)
    rv = make_rv_panel(n=30)

    out = build_vrp_panel(
        iv_df=iv,
        rv_df=rv,
        market="US",
        horizon=3,
        annualization_periods=252,
    )

    for column in [
        "vrp_backward_cc",
        "vrp_backward_parkinson",
        "vrp_backward_rs",
        "vrp_backward_yz",
    ]:
        assert column in out.columns

    from vrp.features.feature_registry import VRP_FEATURE_COLUMNS, VRP_LABEL_COLUMNS

    for column in [
        "vrp_backward_cc",
        "vrp_backward_parkinson",
        "vrp_backward_rs",
        "vrp_backward_yz",
    ]:
        assert column not in VRP_FEATURE_COLUMNS
        assert column not in VRP_LABEL_COLUMNS


def test_forward_expost_remains_gk_only() -> None:
    iv = make_iv_panel(n=30)
    rv = make_rv_panel(n=30)

    out = build_vrp_panel(
        iv_df=iv,
        rv_df=rv,
        market="US",
        horizon=3,
        annualization_periods=252,
    )

    forbidden_columns = [
        "vrp_forward_expost_cc_label",
        "vrp_forward_expost_parkinson_label",
        "vrp_forward_expost_rs_label",
        "vrp_forward_expost_yz_label",
        "rv_yz_daily",
    ]

    for column in forbidden_columns:
        assert column not in out.columns


def test_build_vrp_panel_rejects_wrong_market() -> None:
    iv = make_iv_panel(n=10)
    rv = make_rv_panel(n=10)
    rv["market"] = "INDIA"

    with pytest.raises(ValueError, match="unexpected market"):
        build_vrp_panel(iv_df=iv, rv_df=rv, market="US")