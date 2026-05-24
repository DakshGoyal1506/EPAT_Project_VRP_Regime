# tests/test_no_lookahead.py

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from vrp.features.feature_registry import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    VRP_FEATURE_COLUMNS,
    VRP_LABEL_COLUMNS,
    VRP_ROBUSTNESS_COLUMNS,
    assert_no_lookahead_feature_columns,
    assert_registry_is_valid,
    get_vrp_feature_columns,
    get_vrp_label_columns,
    get_vrp_robustness_columns,
    is_forbidden_feature_column,
)
from vrp.forecasting.har_registry import (
    HAR_FEATURE_COLUMNS,
    HAR_FORBIDDEN_FEATURE_SUBSTRINGS,
    HAR_TARGET_COLUMNS,
    assert_har_registry_is_valid,
)

from vrp.forecasting.har_rv import (
    HARConfig,
    expanding_window_har_forecast,
    fit_har_ols,
    get_available_training_rows,
    prepare_har_model_frame,
)
from vrp.features.vrp import flag_feature_columns_vs_label_columns, compute_har_vrp


from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.regimes.regime_registry import (  # noqa: E402
    assert_no_forbidden_regime_features,
    assert_regime_features_are_point_in_time,
    get_allowed_diagnostic_labels,
    get_allowed_regime_features,
)
from vrp.reports.regime_diagnostics import (  # noqa: E402
    build_threshold_no_lookahead_audit,
)
from .test_threshold_regimes import _base_config, _synthetic_panel  # noqa: E402
from vrp.regimes.threshold import (  # noqa: E402
    classify_all_threshold_components,
    combine_threshold_regimes,
)


def test_regime_registry_rejects_forbidden_regime_columns():
    forbidden_cols = [
        "future_rv",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "regime_label",
    ]

    for col in forbidden_cols:
        with pytest.raises(ValueError):
            assert_no_forbidden_regime_features(["iv_ann", col])

        with pytest.raises(ValueError):
            assert_regime_features_are_point_in_time(["iv_ann", col])


def test_phase3_labels_are_not_allowed_in_threshold_regime_feature_lists():
    diagnostic_labels = get_allowed_diagnostic_labels()
    regime_features = get_allowed_regime_features()

    for label_col in diagnostic_labels:
        assert label_col not in regime_features

        with pytest.raises(ValueError):
            assert_regime_features_are_point_in_time(regime_features + [label_col])


def test_threshold_no_lookahead_audit_has_no_forbidden_columns_for_available_regimes():
    config = _base_config()
    panel = _synthetic_panel(8)

    components = classify_all_threshold_components(panel, config)
    regimes = combine_threshold_regimes(components, config)

    audit = build_threshold_no_lookahead_audit(regimes)

    available_audit = audit[audit["regime_available"]]

    assert not available_audit.empty
    assert available_audit["uses_forbidden_columns"].eq(False).all()
    assert available_audit["uses_strict_prior_thresholds"].eq(True).all()


def test_forward_labels_can_exist_in_panel_but_not_drive_construction():
    config = _base_config()
    panel = _synthetic_panel(8)

    panel["rv_gk_22d_forward_ann_label"] = [1000.0] * len(panel)
    panel["vrp_forward_expost_gk_label"] = [-1000.0] * len(panel)

    panel_without_labels = panel.drop(
        columns=[
            "rv_gk_22d_forward_ann_label",
            "vrp_forward_expost_gk_label",
        ]
    )

    regimes_with_labels = combine_threshold_regimes(
        classify_all_threshold_components(panel, config),
        config,
    )

    regimes_without_labels = combine_threshold_regimes(
        classify_all_threshold_components(panel_without_labels, config),
        config,
    )

    assert regimes_with_labels["threshold_state"].tolist() == regimes_without_labels[
        "threshold_state"
    ].tolist()

    assert regimes_with_labels["threshold_trigger_reason"].tolist() == regimes_without_labels[
        "threshold_trigger_reason"
    ].tolist()


def test_registry_has_no_feature_label_overlap() -> None:
    assert_registry_is_valid()

    overlap = set(VRP_FEATURE_COLUMNS) & set(VRP_LABEL_COLUMNS)
    assert overlap == set()


def test_registry_has_no_feature_robustness_overlap() -> None:
    overlap = set(VRP_FEATURE_COLUMNS) & set(VRP_ROBUSTNESS_COLUMNS)
    assert overlap == set()


def test_feature_columns_do_not_contain_forbidden_substrings() -> None:
    for col in VRP_FEATURE_COLUMNS:
        col_lower = col.lower()
        for token in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert token not in col_lower


def test_label_columns_are_not_features() -> None:
    features = get_vrp_feature_columns()
    labels = get_vrp_label_columns()

    for label in labels:
        assert label not in features


def test_robustness_columns_are_not_primary_features() -> None:
    features = get_vrp_feature_columns()
    labels = get_vrp_label_columns()
    robustness = get_vrp_robustness_columns()

    assert robustness == VRP_ROBUSTNESS_COLUMNS

    for column in robustness:
        assert column not in features
        assert column not in labels


def test_robustness_columns_have_no_forward_or_label_names() -> None:
    for column in VRP_ROBUSTNESS_COLUMNS:
        column_lower = column.lower()
        assert "future" not in column_lower
        assert "forward" not in column_lower
        assert "expost" not in column_lower
        assert "label" not in column_lower


def test_label_columns_have_label_naming() -> None:
    labels = get_vrp_label_columns()

    assert "rv_gk_22d_forward_ann_label" in labels
    assert "vrp_forward_expost_gk_label" in labels

    for label in labels:
        label_lower = label.lower()
        assert "label" in label_lower
        assert ("forward" in label_lower) or ("expost" in label_lower)


def test_is_forbidden_feature_column_detects_bad_names() -> None:
    assert is_forbidden_feature_column("future_rv")
    assert is_forbidden_feature_column("rv_forward")
    assert is_forbidden_feature_column("vrp_expost")
    assert is_forbidden_feature_column("target_label")

    assert not is_forbidden_feature_column("iv_ann")
    assert not is_forbidden_feature_column("rv_gk_22d_ann_lag1")
    assert not is_forbidden_feature_column("vrp_backward_gk")


def test_assert_no_lookahead_feature_columns_rejects_bad_names() -> None:
    with pytest.raises(ValueError, match="forward-looking"):
        assert_no_lookahead_feature_columns(
            [
                "iv_ann",
                "rv_gk_22d_ann_lag1",
                "vrp_forward_expost_gk_label",
            ]
        )


def test_flag_feature_columns_vs_label_columns_separates_features_and_labels() -> None:
    df = pd.DataFrame(
        {
            "iv_ann": [0.04, 0.05, None],
            "rv_gk_22d_ann_lag1": [0.03, 0.04, 0.05],
            "vrp_backward_gk": [0.01, 0.01, None],
            "vrp_backward_gk_positive": [True, True, None],
            "rv_gk_22d_forward_ann_label": [0.05, None, None],
            "vrp_forward_expost_gk_label": [-0.01, None, None],
        }
    )

    out = flag_feature_columns_vs_label_columns(df)

    assert "feature_allowed" in out.columns
    assert bool(out.loc[0, "feature_allowed"])
    assert bool(out.loc[1, "feature_allowed"])
    assert not bool(out.loc[2, "feature_allowed"])


def test_flag_feature_columns_vs_label_columns_requires_labels_present() -> None:
    df = pd.DataFrame(
        {
            "iv_ann": [0.04],
            "rv_gk_22d_ann_lag1": [0.03],
            "vrp_backward_gk": [0.01],
            "vrp_backward_gk_positive": [True],
        }
    )

    with pytest.raises(ValueError, match="label"):
        flag_feature_columns_vs_label_columns(df)


def test_flag_feature_columns_vs_label_columns_requires_features_present() -> None:
    df = pd.DataFrame(
        {
            "rv_gk_22d_forward_ann_label": [0.05],
            "vrp_forward_expost_gk_label": [-0.01],
        }
    )

    with pytest.raises(ValueError, match="feature"):
        flag_feature_columns_vs_label_columns(df)


# def test_har_registry_and_audit_behaviour() -> None:
#     """
#     Basic HAR registry and audit checks:
#     - HAR feature set is exactly the three approved lagged predictors
#     - `fit_har_ols` rejects forbidden predictor lists (e.g. containing `iv_ann` or forward label)
#     - Audit rows report max_training_target_end_date < forecast_date when training is present
#     """
#     from vrp.forecasting.har_registry import HAR_FEATURE_COLUMNS, assert_primary_har_features
#     from vrp.forecasting.har_rv import (
#         fit_har_ols,
#         expanding_window_har_forecast,
#         load_har_config,
#     )

#     # registry exactness
#     assert HAR_FEATURE_COLUMNS == [
#         "har_rv_d_lag1_ann",
#         "har_rv_w_lag1_ann",
#         "har_rv_m_lag1_ann",
#     ]
#     assert_primary_har_features(list(HAR_FEATURE_COLUMNS))

#     # fit_har_ols should reject forbidden predictor lists via validate_primary_har_feature_cols
#     dummy = pd.DataFrame({
#         "date": pd.date_range("2020-01-01", periods=10, freq="D"),
#         "har_rv_d_lag1_ann": np.linspace(0.01, 0.02, 10),
#         "har_rv_w_lag1_ann": np.linspace(0.01, 0.02, 10),
#         "har_rv_m_lag1_ann": np.linspace(0.01, 0.02, 10),
#         "iv_ann": np.linspace(0.03, 0.04, 10),
#         "rv_gk_22d_forward_ann_label": np.linspace(0.02, 0.03, 10),
#     })

#     # Wrong feature list with iv_ann present
#     with pytest.raises(ValueError):
#         fit_har_ols(dummy, ["iv_ann", "har_rv_w_lag1_ann", "har_rv_m_lag1_ann"], "rv_gk_22d_forward_ann_label")

#     # Wrong feature list with forward label present
#     with pytest.raises(ValueError):
#         fit_har_ols(dummy, ["har_rv_d_lag1_ann", "har_rv_w_lag1_ann", "rv_gk_22d_forward_ann_label"], "rv_gk_22d_forward_ann_label")

#     # Audit behaviour: run quick expanding forecast and check audit invariant
#     panel = pd.DataFrame({
#         "date": pd.date_range("2020-01-01", periods=40, freq="D"),
#         "market": ["US"] * 40,
#         "rv_gk_daily": 0.0001 + 0.00001 * np.arange(40),
#     })
#     # build har features and forward label using existing helpers
#     from vrp.forecasting.har_rv import make_har_features, add_forward_target_metadata, load_har_config

#     panel = make_har_features(panel, daily_rv_col="rv_gk_daily", horizon=22, annualization_periods=252)
#     # build forward target using strictly future daily RV values
#     horizon = 22
#     future_cols = [panel["rv_gk_daily"].shift(-i) for i in range(1, horizon + 1)]
#     future_stack = pd.concat(future_cols, axis=1)
#     valid = future_stack.notna().sum(axis=1) == horizon
#     panel["rv_gk_22d_forward_ann_label"] = (252 * future_stack.mean(axis=1)).where(valid)
#     # add a simple positive IV column required by prepare_har_model_frame
#     # use monthly HAR feature as a proxy plus a small offset to ensure positivity
#     panel["iv_ann"] = panel["har_rv_m_lag1_ann"].fillna(0.0) + 0.00005
#     panel = add_forward_target_metadata(panel, horizon=22, target_col="rv_gk_22d_forward_ann_label")

#     cfg = load_har_config(Path("configs/har_rv.yaml")).model_copy(update={
#         "min_train_observations": 5,
#         "rolling_train_window": 10,
#         "forecast_horizon": 22,
#     })

#     f, c, a = expanding_window_har_forecast(panel.copy(), cfg)
#     if len(a) > 0:
#         # where there is a max_training_target_end_date present, it must be strictly before forecast_date
#         for _, r in a.iterrows():
#             if pd.notna(r.get("max_training_target_end_date")):
#                 max_te = pd.to_datetime(r["max_training_target_end_date"], errors="coerce")
#                 fdate = pd.to_datetime(r["forecast_date"], errors="coerce")
#                 assert pd.notna(max_te) and pd.notna(fdate)
#                 assert max_te < fdate

HAR_TEST_TARGET_COL = "rv_gk_22d_forward_ann_label"


def _make_forward_label_from_daily(
    daily_rv: pd.Series,
    *,
    horizon: int,
    annualization_periods: int = 252,
) -> pd.Series:
    """
    Strict forward target:
        annualization_periods * mean(rv_daily[t+1], ..., rv_daily[t+horizon])
    """
    future = pd.concat(
        [daily_rv.shift(-step) for step in range(1, horizon + 1)],
        axis=1,
    )
    valid = future.notna().sum(axis=1) == horizon
    label = annualization_periods * future.mean(axis=1)
    return label.where(valid)


def _make_har_no_lookahead_panel(
    *,
    n: int = 90,
    horizon: int = 3,
) -> pd.DataFrame:
    """
    Synthetic Phase 3-like panel for HAR no-lookahead tests.
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    daily = pd.Series(np.arange(1, n + 1, dtype=float) / 252.0)

    panel = pd.DataFrame(
        {
            "date": dates,
            "market": "US",
            "iv_ann": 0.04 + 0.0001 * np.arange(n, dtype=float),
            "iv_close": 100.0 * np.sqrt(0.04 + 0.0001 * np.arange(n, dtype=float)),
            "rv_gk_daily": daily,
        }
    )

    panel["rv_gk_22d_ann"] = 252.0 * daily.rolling(22, min_periods=22).mean()
    panel["rv_gk_22d_ann_lag1"] = panel["rv_gk_22d_ann"].shift(1)

    panel[HAR_TEST_TARGET_COL] = _make_forward_label_from_daily(
        panel["rv_gk_daily"],
        horizon=horizon,
        annualization_periods=252,
    )

    panel["vrp_backward_gk"] = panel["iv_ann"] - panel["rv_gk_22d_ann_lag1"]
    panel["vrp_backward_gk_positive"] = (
        panel["vrp_backward_gk"] > 0
    ).where(panel["vrp_backward_gk"].notna(), pd.NA).astype("boolean")
    panel["vrp_forward_expost_gk_label"] = panel["iv_ann"] - panel[HAR_TEST_TARGET_COL]
    panel["feature_allowed"] = panel[
        ["iv_ann", "rv_gk_22d_ann_lag1", "vrp_backward_gk"]
    ].notna().all(axis=1)

    return panel


def test_har_registry_is_valid_and_features_have_no_forbidden_substrings() -> None:
    assert_har_registry_is_valid()

    assert HAR_FEATURE_COLUMNS == [
        "har_rv_d_lag1_ann",
        "har_rv_w_lag1_ann",
        "har_rv_m_lag1_ann",
    ]

    for column in HAR_FEATURE_COLUMNS:
        column_lower = column.lower()
        for token in HAR_FORBIDDEN_FEATURE_SUBSTRINGS:
            assert token not in column_lower


def test_har_targets_are_label_columns_not_features() -> None:
    assert HAR_TARGET_COLUMNS == ["rv_gk_22d_forward_ann_label"]

    for target in HAR_TARGET_COLUMNS:
        assert "label" in target.lower()
        assert target not in HAR_FEATURE_COLUMNS


def test_har_model_rejects_iv_ann_as_predictor() -> None:
    horizon = 3
    panel = _make_har_no_lookahead_panel(n=80, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    with pytest.raises(ValueError, match="Primary HAR predictors"):
        fit_har_ols(
            frame,
            feature_cols=[
                "har_rv_d_lag1_ann",
                "har_rv_w_lag1_ann",
                "iv_ann",
            ],
            target_col=cfg.target_col,
            hac_maxlags=0,
        )


def test_har_model_rejects_forward_label_as_predictor() -> None:
    horizon = 3
    panel = _make_har_no_lookahead_panel(n=80, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        compute_backend="cpu_statsmodels",
        coefficient_hac_frequency="none",
    )

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    with pytest.raises(ValueError, match="Primary HAR predictors"):
        fit_har_ols(
            frame,
            feature_cols=[
                "har_rv_d_lag1_ann",
                "har_rv_w_lag1_ann",
                HAR_TEST_TARGET_COL,
            ],
            target_col=cfg.target_col,
            hac_maxlags=0,
        )


def test_har_available_training_rows_exclude_overlapping_forward_windows() -> None:
    horizon = 3
    panel = _make_har_no_lookahead_panel(n=80, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        compute_backend="cpu_numpy_batched",
        coefficient_hac_frequency="none",
    )

    frame = prepare_har_model_frame(panel, config=cfg, validate_target=True)

    forecast_idx = 40
    forecast_date = frame.loc[forecast_idx, "date"]

    candidate_df, valid_df = get_available_training_rows(
        df=frame,
        forecast_date=forecast_date,
        feature_cols=cfg.feature_cols,
        target_col=cfg.target_col,
        config=cfg,
    )

    assert not candidate_df.empty
    assert not valid_df.empty

    assert (valid_df["date"] < forecast_date).all()
    assert (valid_df["target_end_date"] < forecast_date).all()

    overlapping = candidate_df[
        candidate_df["target_end_date"] >= forecast_date
    ]
    assert len(overlapping) > 0

    valid_dates = set(valid_df["date"])
    overlapping_dates = set(overlapping["date"])
    assert valid_dates.isdisjoint(overlapping_dates)


def test_har_forecast_audit_passes_for_available_rows() -> None:
    horizon = 3
    panel = _make_har_no_lookahead_panel(n=100, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        compute_backend="cpu_numpy_batched",
        coefficient_hac_frequency="none",
    )

    forecast, coefficients, audit = expanding_window_har_forecast(panel, cfg)

    assert not forecast.empty
    assert not audit.empty
    assert not coefficients.empty

    available = audit[audit["forecast_available"].astype(bool)].copy()
    assert not available.empty

    available["forecast_date"] = pd.to_datetime(available["forecast_date"])
    available["max_training_target_end_date"] = pd.to_datetime(
        available["max_training_target_end_date"]
    )

    bad = available[
        available["max_training_target_end_date"] >= available["forecast_date"]
    ]

    assert bad.empty
    assert available["rule_target_end_before_forecast_date"].astype(bool).all()


def test_har_forecast_panel_keeps_forward_label_as_label_only() -> None:
    horizon = 3
    panel = _make_har_no_lookahead_panel(n=100, horizon=horizon)

    cfg = HARConfig(
        forecast_horizon=horizon,
        min_train_observations=20,
        compute_backend="cpu_numpy_batched",
        coefficient_hac_frequency="none",
    )

    forecast, _coefficients, _audit = expanding_window_har_forecast(panel, cfg)

    assert HAR_TEST_TARGET_COL in forecast.columns

    for feature_col in HAR_FEATURE_COLUMNS:
        assert feature_col in forecast.columns
        assert "forward" not in feature_col.lower()
        assert "label" not in feature_col.lower()

    assert HAR_TEST_TARGET_COL not in HAR_FEATURE_COLUMNS


def test_vrp_har_is_future_phase_feature_only_when_forecast_available() -> None:
    vrp = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "market": ["US"] * 5,
            "iv_ann": [0.10, 0.20, 0.30, 0.40, 0.50],
            "rv_gk_daily": [0.01] * 5,
            "rv_gk_22d_ann_lag1": [np.nan, 0.08, 0.09, 0.10, 0.11],
            "vrp_backward_gk": [np.nan, 0.12, 0.21, 0.30, 0.39],
            "vrp_backward_gk_positive": [pd.NA, True, True, True, True],
            HAR_TEST_TARGET_COL: [0.11, 0.12, 0.13, np.nan, np.nan],
            "vrp_forward_expost_gk_label": [-0.01, 0.08, 0.17, np.nan, np.nan],
            "feature_allowed": [False, True, True, True, True],
        }
    )

    har = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "market": ["US"] * 5,
            "har_rv_gk_22d_forecast_ann": [np.nan, 0.15, np.nan, 0.25, np.nan],
            "har_forecast_available": [False, True, False, True, False],
            "har_blocked_reason": [
                "insufficient_training_history",
                pd.NA,
                "missing_har_features",
                pd.NA,
                "missing_target_metadata",
            ],
        }
    )

    out = compute_har_vrp(vrp, har)

    unavailable = ~out["har_forecast_available"].astype(bool)

    assert out.loc[unavailable, "vrp_har_gk"].isna().all()
    assert out.loc[~unavailable, "vrp_har_gk"].notna().all()

    assert out.loc[1, "vrp_har_gk"] == pytest.approx(0.20 - 0.15)
    assert out.loc[3, "vrp_har_gk"] == pytest.approx(0.40 - 0.25)


def test_har_forward_label_remains_forbidden_as_live_feature_name() -> None:
    bad_live_features = [
        "iv_ann",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
    ]

    for column in bad_live_features:
        if column in HAR_FEATURE_COLUMNS:
            raise AssertionError(f"{column} must not be a HAR live feature")

    assert "rv_gk_22d_forward_ann_label" in HAR_TARGET_COLUMNS
    

# ---------------------------------------------------------------------------
# Phase 6 HMM no-lookahead integration checks
# ---------------------------------------------------------------------------


def test_hmm_registry_rejects_forward_threshold_crisis_and_hmm_derived_features() -> None:
    from vrp.regimes.hmm_validation import assert_hmm_feature_columns_are_legal

    forbidden = [
        "future_rv",
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "threshold_state",
        "threshold_trigger_reason",
        "crisis_window_flag",
        "hmm_filtered_prob_stress",
    ]

    for col in forbidden:
        with pytest.raises(ValueError):
            assert_hmm_feature_columns_are_legal(["iv_ann", col])


def test_hmm_probability_policy_forbids_smoothed_probabilities_for_backtest() -> None:
    from vrp.regimes.hmm_validation import assert_output_probability_policy_is_safe

    assert_output_probability_policy_is_safe(
        uses_custom_forward_filter=True,
        uses_hmmlearn_predict_proba_for_backtest=False,
        uses_smoothed_probabilities_for_backtest=False,
    )

    with pytest.raises(ValueError):
        assert_output_probability_policy_is_safe(
            uses_custom_forward_filter=True,
            uses_hmmlearn_predict_proba_for_backtest=False,
            uses_smoothed_probabilities_for_backtest=True,
        )

    with pytest.raises(ValueError):
        assert_output_probability_policy_is_safe(
            uses_custom_forward_filter=True,
            uses_hmmlearn_predict_proba_for_backtest=True,
            uses_smoothed_probabilities_for_backtest=False,
        )


def test_hmm_crisis_windows_and_threshold_states_are_diagnostic_only() -> None:
    from vrp.regimes.hmm_validation import (
        validate_crisis_windows_usage,
        validate_threshold_comparison_usage,
    )

    validate_crisis_windows_usage(
        used_for="crisis_stress_overlap",
        diagnostics_only=True,
    )

    validate_crisis_windows_usage(
        used_for="crisis_lead_lag",
        diagnostics_only=True,
    )

    validate_threshold_comparison_usage(
        threshold_state_as_feature=False,
        threshold_state_as_target=False,
        choose_model_by_threshold_match=False,
    )

    with pytest.raises(ValueError):
        validate_crisis_windows_usage(
            used_for="model_selection",
            diagnostics_only=True,
        )

    with pytest.raises(ValueError):
        validate_threshold_comparison_usage(
            threshold_state_as_feature=True,
        )


def test_hmm_signal_columns_encode_next_session_usage() -> None:
    from vrp.regimes.hmm_features import build_hmm_feature_panel
    from vrp.regimes.hmm_scaling import scale_hmm_feature_panel
    from vrp.regimes.gaussian_hmm import (
        HMMCandidateSpec,
        HMMFitConfig,
        fit_and_build_hmm_candidate_output,
    )

    rng = np.random.default_rng(123)

    regime = np.tile(np.r_[np.zeros(60), np.ones(60)], 10).astype(int)
    n = len(regime)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "vrp_har_gk": np.where(
                regime == 0,
                rng.normal(0.04, 0.01, n),
                rng.normal(-0.02, 0.02, n),
            ),
            "rv_gk_22d_ann_lag1": np.where(
                regime == 0,
                rng.normal(0.10, 0.02, n),
                rng.normal(0.35, 0.05, n),
            ),
            "iv_ann": np.where(
                regime == 0,
                rng.normal(0.15, 0.02, n),
                rng.normal(0.45, 0.05, n),
            ),
            "log_return": np.where(
                regime == 0,
                rng.normal(0.001, 0.005, n),
                rng.normal(-0.002, 0.02, n),
            ),
            "har_forecast_available": True,
        }
    )

    feature_panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    scaled = scale_hmm_feature_panel(
        feature_panel,
        train_fraction=0.70,
        min_train_observations=750,
        min_test_observations=250,
    )

    output = fit_and_build_hmm_candidate_output(
        scaled,
        spec=HMMCandidateSpec("F3", 2, "diag"),
        fit_config=HMMFitConfig(
            n_init=2,
            n_iter=150,
            random_seed=42,
        ),
    )

    assert output.output_panel is not None

    panel = output.output_panel.copy()

    required = [
        "hmm_signal_observation_date",
        "hmm_signal_available_after_close_date",
        "hmm_signal_trade_date",
        "hmm_state_for_next_session",
        "hmm_state_name_for_next_session",
        "hmm_filtered_prob_calm_for_next_session",
        "hmm_filtered_prob_transition_for_next_session",
        "hmm_filtered_prob_stress_for_next_session",
    ]

    for col in required:
        assert col in panel.columns

    dates = pd.to_datetime(panel["date"])
    trade_dates = pd.to_datetime(panel["hmm_signal_trade_date"], errors="coerce")
    usable = trade_dates.notna()

    assert usable.any()
    assert (trade_dates.loc[usable] > dates.loc[usable]).all()
    assert pd.isna(panel.loc[len(panel) - 1, "hmm_signal_trade_date"])

import json
from pathlib import Path

import pandas as pd
import pytest

from vrp.backtest.backtest_config import load_backtest_config
from vrp.backtest.final_audit import run_phase10_final_audit
from vrp.backtest.payoff_proxies import (
    PRIMARY_PAYOFF_LABEL,
    build_forward_vrp_outcome_panel,
    compute_forward_vrp_strategy_payoff,
    join_strategy_with_outcome,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CONFIG_PATH = REPO_ROOT / "configs" / "backtest.yaml"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_toy_signals() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")


def _load_toy_outcomes() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")


def test_phase10_outcome_label_not_present_in_raw_toy_signals() -> None:
    signals = _load_toy_signals()

    assert PRIMARY_PAYOFF_LABEL not in signals.columns


def test_phase10_outcome_label_appears_only_after_outcome_join() -> None:
    signals = _load_toy_signals()
    outcomes = build_forward_vrp_outcome_panel(_load_toy_outcomes())

    joined = join_strategy_with_outcome(signals, outcomes)

    assert PRIMARY_PAYOFF_LABEL in joined.columns


def test_phase10_default_join_uses_signal_date_not_trade_date() -> None:
    signals = pd.DataFrame(
        {
            "market": ["US"],
            "strategy_name": ["unconditional_full"],
            "signal_observation_date": ["2020-01-02"],
            "target_trade_date": ["2020-01-03"],
            "target_exposure": [-1.0],
            "strategy_available": [True],
        }
    )

    outcomes = pd.DataFrame(
        {
            "market": ["US", "US"],
            "date": ["2020-01-02", "2020-01-03"],
            PRIMARY_PAYOFF_LABEL: [0.04, -0.99],
        }
    )

    outcome_panel = build_forward_vrp_outcome_panel(outcomes)
    joined = join_strategy_with_outcome(
        signals,
        outcome_panel,
        alignment="signal_observation_date",
    )
    payoff = compute_forward_vrp_strategy_payoff(joined)

    row = payoff.iloc[0]

    assert row["outcome_label_date"] == pd.Timestamp("2020-01-02")
    assert float(row[PRIMARY_PAYOFF_LABEL]) == 0.04
    assert float(row["gross_return_proxy"]) == 0.04


def test_phase10_eligible_rows_trade_after_signal_in_real_outputs_if_present() -> None:
    us_panel = REPO_ROOT / "data" / "processed" / "us_backtest_panel.parquet"
    india_panel = REPO_ROOT / "data" / "processed" / "india_backtest_panel.parquet"

    missing = [path for path in [us_panel, india_panel] if not path.exists()]
    if missing:
        pytest.skip(f"Phase 10 real backtest panels not present: {missing}")

    for path in [us_panel, india_panel]:
        panel = pd.read_parquet(path)
        eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

        signal_dates = pd.to_datetime(
            panel.loc[eligible, "signal_observation_date"],
            errors="coerce",
        )
        trade_dates = pd.to_datetime(
            panel.loc[eligible, "target_trade_date"],
            errors="coerce",
        )
        outcome_dates = pd.to_datetime(
            panel.loc[eligible, "outcome_label_date"],
            errors="coerce",
        )

        assert (trade_dates > signal_dates).all()
        assert (outcome_dates == signal_dates).all()


def test_phase10_real_metadata_has_zero_no_lookahead_violations_if_present() -> None:
    metadata_paths = [
        REPO_ROOT / "data" / "processed" / "us_backtest_panel_metadata.json",
        REPO_ROOT / "data" / "processed" / "india_backtest_panel_metadata.json",
    ]

    missing = [path for path in metadata_paths if not path.exists()]
    if missing:
        pytest.skip(f"Phase 10 metadata sidecars not present: {missing}")

    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["n_target_not_after_signal_violations"] == 0
        assert payload["n_outcome_not_equal_signal_date_violations"] == 0
        assert payload["label_role"] == "realised_outcome_only"
        assert payload["research_proxy_not_trade_pnl"] is True
        assert payload["overlapping_labels"] is True


def test_phase10_final_audit_passes_on_real_outputs_if_present() -> None:
    required = [
        REPO_ROOT / "configs" / "backtest.yaml",
        REPO_ROOT / "data" / "processed" / "us_backtest_panel.parquet",
        REPO_ROOT / "data" / "processed" / "india_backtest_panel.parquet",
        REPO_ROOT / "reports" / "tables" / "phase_10" / "backtest_metadata.json",
        REPO_ROOT / "reports" / "tables" / "phase_10" / "robustness_metadata.json",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"Phase 10 real outputs not present: {missing}")

    config = load_backtest_config(BACKTEST_CONFIG_PATH)
    result = run_phase10_final_audit(
        config=config,
        repo_root=REPO_ROOT,
        market="ALL",
        require_robustness=True,
    )

    assert result.status == "passed"
    assert result.n_errors == 0