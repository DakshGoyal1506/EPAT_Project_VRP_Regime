# tests/test_har_rv.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vrp.features.vrp import compute_har_vrp
from vrp.forecasting.har_registry import HAR_FEATURE_COLUMNS
from vrp.forecasting.har_rv import (
    HARConfig,
    add_forward_target_metadata,
    expanding_window_har_forecast,
    fit_har_ols,
    get_available_training_rows,
    make_har_features,
    prepare_har_model_frame,
    recompute_forward_target_for_validation,
    validate_phase3_target_matches_recomputed_target,
)


TARGET_COL = "rv_gk_22d_forward_ann_label"


def make_forward_label_from_daily(
    daily_rv: pd.Series,
    *,
    horizon: int,
    annualization_periods: int = 252,
) -> pd.Series:
    """
    Test helper matching the Phase 3 forward-label convention.

    target_t =
        annualization_periods * mean(rv_daily[t+1], ..., rv_daily[t+horizon])
    """
    future = pd.concat(
        [daily_rv.shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    )

    valid = future.notna().sum(axis=1) == horizon
    label = annualization_periods * future.mean(axis=1)

    return label.where(valid)


def make_synthetic_har_panel(
    *,
    n: int = 80,
    horizon: int = 3,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    Build a one-market synthetic Phase 3-like VRP panel.

    rv_gk_daily values are deterministic:
        [1, 2, ..., n] / 252

    The forward label uses strictly future values only.
    """
    dates = pd.date_range(start=start, periods=n, freq="D")
    daily_rv = pd.Series(
        np.arange(1, n + 1, dtype=float) / 252.0,
        index=range(n),
    )

    panel = pd.DataFrame(
        {
            "date": dates,
            "market": "US",
            "iv_ann": 0.05 + np.arange(n, dtype=float) * 0.0001,
            "iv_close": 100.0 * np.sqrt(0.05 + np.arange(n, dtype=float) * 0.0001),
            "rv_gk_daily": daily_rv,
            "rv_gk_22d_ann": (
                252.0
                * daily_rv.rolling(window=22, min_periods=22).mean()
            ),
        }
    )

    panel["rv_gk_22d_ann_lag1"] = panel["rv_gk_22d_ann"].shift(1)
    panel[TARGET_COL] = make_forward_label_from_daily(
        panel["rv_gk_daily"],
        horizon=horizon,
        annualization_periods=252,
    )
    panel["vrp_backward_gk"] = panel["iv_ann"] - panel["rv_gk_22d_ann_lag1"]
    panel["vrp_backward_gk_positive"] = (
        panel["vrp_backward_gk"] > 0
    ).where(panel["vrp_backward_gk"].notna(), None).astype("boolean")
    panel["vrp_forward_expost_gk_label"] = panel["iv_ann"] - panel[TARGET_COL]
    panel["feature_allowed"] = panel[
        ["iv_ann", "rv_gk_22d_ann_lag1", "vrp_backward_gk"]
    ].notna().all(axis=1)

    return panel


def test_har_feature_lagging_daily_weekly_monthly() -> None:
    """
    HAR features must use only realised variance through t-1.
    """
    n = 40
    horizon = 22

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "market": "US",
            "rv_gk_daily": np.arange(1, n + 1, dtype=float) / 252.0,
        }
    )

    out = make_har_features(
        panel,
        daily_rv_col="rv_gk_daily",
        horizon=horizon,
        annualization_periods=252,
    )

    row = 25

    expected_daily = 252.0 * panel["rv_gk_daily"].iloc[row - 1]
    expected_weekly = 252.0 * panel.loc[row - 5 : row - 1, "rv_gk_daily"].mean()
    expected_monthly = 252.0 * panel.loc[row - 22 : row - 1, "rv_gk_daily"].mean()

    assert out.loc[row, "har_rv_d_lag1_ann"] == pytest.approx(expected_daily)
    assert out.loc[row, "har_rv_w_lag1_ann"] == pytest.approx(expected_weekly)
    assert out.loc[row, "har_rv_m_lag1_ann"] == pytest.approx(expected_monthly)

    assert out.loc[row, "har_rv_d_lag1_ann"] == pytest.approx(float(row))
    assert out.loc[row, "har_rv_w_lag1_ann"] == pytest.approx(np.mean([21, 22, 23, 24, 25]))
    assert out.loc[row, "har_rv_m_lag1_ann"] == pytest.approx(np.mean(np.arange(4, 26)))


def test_har_feature_lagging_has_expected_initial_nans() -> None:
    """
    Early rows must be NaN where lagged history is incomplete.
    """
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "market": "US",
            "rv_gk_daily": np.arange(1, 31, dtype=float) / 252.0,
        }
    )

    out = make_har_features(panel, horizon=22)

    assert pd.isna(out.loc[0, "har_rv_d_lag1_ann"])
    assert pd.isna(out.loc[4, "har_rv_w_lag1_ann"])
    assert pd.notna(out.loc[5, "har_rv_w_lag1_ann"])
    assert pd.isna(out.loc[21, "har_rv_m_lag1_ann"])
    assert pd.notna(out.loc[22, "har_rv_m_lag1_ann"])


def test_forward_target_timing_recomputation_uses_strictly_future_values() -> None:
    """
    Validation recomputation must use t+1 ... t+horizon only.
    """
    horizon = 3

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "market": "US",
            "rv_gk_daily": np.arange(1, 11, dtype=float) / 252.0,
        }
    )

    out = recompute_forward_target_for_validation(
        panel,
        daily_rv_col="rv_gk_daily",
        horizon=horizon,
        annualization_periods=252,
        output_col="_target",
    )

    # row 0 uses values 2, 3, 4; not 1, not 5.
    assert out.loc[0, "_target"] == pytest.approx(np.mean([2, 3, 4]))

    # row 4 uses values 6, 7, 8.
    assert out.loc[4, "_target"] == pytest.approx(np.mean([6, 7, 8]))

    # last horizon rows lack complete future window.
    assert out["_target"].tail(horizon).isna().all()


def test_target_start_and_end_dates() -> None:
    """
    target_start_date and target_end_date must identify the forward target window.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=10, horizon=horizon)

    out = add_forward_target_metadata(
        panel,
        horizon=horizon,
        target_col=TARGET_COL,
    )

    assert out.loc[0, "target_start_date"] == panel.loc[1, "date"]
    assert out.loc[0, "target_end_date"] == panel.loc[3, "date"]

    assert out.loc[5, "target_start_date"] == panel.loc[6, "date"]
    assert out.loc[5, "target_end_date"] == panel.loc[8, "date"]

    assert pd.isna(out.loc[len(out) - 1, "target_start_date"])
    assert pd.isna(out.loc[len(out) - horizon, "target_end_date"])


def test_phase3_target_validation_matches_recomputed_target() -> None:
    """
    Existing Phase 3 target should match validation-only recomputation.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=30, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=5,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    result = validate_phase3_target_matches_recomputed_target(panel, config=cfg)

    assert result["passed"] is True
    assert result["n_compared"] == 30 - horizon


def test_phase3_target_validation_fails_on_wrong_target() -> None:
    """
    A wrong target must fail loudly.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=30, horizon=horizon)

    bad = panel.copy()
    bad.loc[0, TARGET_COL] = bad.loc[0, TARGET_COL] + 1.0

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=5,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_phase3_target_matches_recomputed_target(bad, config=cfg)


def test_prepare_har_model_frame_adds_features_and_target_metadata() -> None:
    """
    prepare_har_model_frame should produce all required modelling columns.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=40, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=5,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    out = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    required = [
        "har_rv_d_lag1_ann",
        "har_rv_w_lag1_ann",
        "har_rv_m_lag1_ann",
        "target_col",
        "target_start_date",
        "target_end_date",
    ]

    for col in required:
        assert col in out.columns

    assert (out["target_col"] == TARGET_COL).all()


def test_training_rows_require_target_end_before_forecast_date() -> None:
    """
    Training rows for forecast date t must satisfy target_end_date_s < t.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=60, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=10,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    forecast_row = 30
    forecast_date = frame.loc[forecast_row, "date"]

    candidate_df, valid_df = get_available_training_rows(
        df=frame,
        forecast_date=forecast_date,
        feature_cols=cfg.feature_cols,
        target_col=cfg.target_col,
        config=cfg,
    )

    assert len(candidate_df) > len(valid_df)
    assert len(valid_df) > 0

    assert (valid_df["target_end_date"] < forecast_date).all()
    assert (valid_df["date"] < forecast_date).all()

    overlapping_candidate = candidate_df[
        candidate_df["target_end_date"] >= forecast_date
    ]
    assert len(overlapping_candidate) > 0


def test_insufficient_history_blocks_forecast() -> None:
    """
    If available training rows are fewer than min_train_observations,
    forecast must be NaN and blocked.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=50, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=10_000,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    forecast, coefficients, audit = expanding_window_har_forecast(panel, cfg)

    assert forecast["har_forecast_available"].sum() == 0
    assert forecast["har_rv_gk_22d_forecast_ann"].isna().all()
    assert (
        forecast["har_blocked_reason"].dropna() == "insufficient_training_history"
    ).any()

    assert coefficients.empty
    assert not audit.empty


def test_forecast_positivity_after_floor() -> None:
    """
    Available HAR forecasts must be positive after forecast_floor.
    """
    horizon = 3
    panel = make_synthetic_har_panel(n=90, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        forecast_floor=1.0e-8,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    forecast, _coefficients, audit = expanding_window_har_forecast(panel, cfg)

    available = forecast["har_forecast_available"].astype(bool)

    assert available.any()
    assert (
        forecast.loc[available, "har_rv_gk_22d_forecast_ann"] >= cfg.forecast_floor
    ).all()

    available_audit = audit[audit["forecast_available"].astype(bool)]
    assert not available_audit.empty
    assert available_audit["rule_target_end_before_forecast_date"].astype(bool).all()


def test_primary_har_rejects_iv_ann_as_predictor() -> None:
    """
    Primary HAR must not use implied variance as a predictor.
    """
    panel = make_synthetic_har_panel(n=60, horizon=3)
    cfg = HARConfig(forecast_horizon=3, min_train_observations=10)

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    with pytest.raises(ValueError, match="Primary HAR predictors"):
        fit_har_ols(
            frame,
            feature_cols=["iv_ann", "har_rv_d_lag1_ann"],
            target_col=TARGET_COL,
            hac_maxlags=0,
        )


def test_primary_har_rejects_label_column_as_predictor() -> None:
    """
    Primary HAR must not use label columns as predictors.
    """
    panel = make_synthetic_har_panel(n=60, horizon=3)
    cfg = HARConfig(forecast_horizon=3, min_train_observations=10)

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    with pytest.raises(ValueError, match="Primary HAR predictors"):
        fit_har_ols(
            frame,
            feature_cols=[
                "har_rv_d_lag1_ann",
                "har_rv_w_lag1_ann",
                TARGET_COL,
            ],
            target_col=TARGET_COL,
            hac_maxlags=0,
        )


def test_primary_har_accepts_exact_registered_predictors() -> None:
    """
    Primary HAR should fit when predictors equal HAR_FEATURE_COLUMNS exactly.
    """
    panel = make_synthetic_har_panel(n=80, horizon=3)
    cfg = HARConfig(forecast_horizon=3, min_train_observations=20)

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    candidate_df, valid_df = get_available_training_rows(
        df=frame,
        forecast_date=frame.loc[50, "date"],
        feature_cols=HAR_FEATURE_COLUMNS,
        target_col=TARGET_COL,
        config=cfg,
    )

    assert not candidate_df.empty
    assert len(valid_df) >= 20

    result = fit_har_ols(
        valid_df,
        feature_cols=HAR_FEATURE_COLUMNS,
        target_col=TARGET_COL,
        hac_maxlags=0,
    )

    assert result is not None
    assert "const" in result.params.index


def test_compute_har_vrp_is_conditional_on_forecast_available() -> None:
    """
    vrp_har_gk must be non-null only when har_forecast_available is true.
    """
    vrp = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "market": ["US", "US", "US", "US"],
            "iv_ann": [0.10, 0.20, 0.30, 0.40],
            "rv_gk_daily": [0.01, 0.01, 0.01, 0.01],
            "rv_gk_22d_ann_lag1": [np.nan, 0.08, 0.09, 0.10],
            "vrp_backward_gk": [np.nan, 0.12, 0.21, 0.30],
            "vrp_backward_gk_positive": [pd.NA, True, True, True],
            TARGET_COL: [0.11, 0.12, np.nan, np.nan],
            "vrp_forward_expost_gk_label": [-0.01, 0.08, np.nan, np.nan],
            "feature_allowed": [False, True, True, True],
        }
    )

    har = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "market": ["US", "US", "US", "US"],
            "har_rv_gk_22d_forecast_ann": [np.nan, 0.15, np.nan, 0.25],
            "har_forecast_available": [False, True, False, True],
            "har_blocked_reason": [
                "insufficient_training_history",
                pd.NA,
                "missing_har_features",
                pd.NA,
            ],
        }
    )

    out = compute_har_vrp(vrp, har)

    assert pd.isna(out.loc[0, "vrp_har_gk"])
    assert out.loc[1, "vrp_har_gk"] == pytest.approx(0.20 - 0.15)
    assert pd.isna(out.loc[2, "vrp_har_gk"])
    assert out.loc[3, "vrp_har_gk"] == pytest.approx(0.40 - 0.25)

    unavailable = ~out["har_forecast_available"].astype(bool)
    assert out.loc[unavailable, "vrp_har_gk"].isna().all()

    assert out.loc[1, "vrp_har_gk_positive"] is True or bool(
        out.loc[1, "vrp_har_gk_positive"]
    )
    assert out.loc[3, "vrp_har_gk_positive"] is True or bool(
        out.loc[3, "vrp_har_gk_positive"]
    )


def test_compute_har_vrp_rejects_existing_har_columns_in_phase3_panel() -> None:
    """
    compute_har_vrp should reject already-joined HAR-VRP panels.
    """
    vrp = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "market": ["US", "US"],
            "iv_ann": [0.10, 0.20],
            "har_rv_gk_22d_forecast_ann": [0.08, 0.09],
        }
    )

    har = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "market": ["US", "US"],
            "har_rv_gk_22d_forecast_ann": [0.08, 0.09],
            "har_forecast_available": [True, True],
            "har_blocked_reason": [pd.NA, pd.NA],
        }
    )

    with pytest.raises(ValueError, match="already contains HAR output"):
        compute_har_vrp(vrp, har)