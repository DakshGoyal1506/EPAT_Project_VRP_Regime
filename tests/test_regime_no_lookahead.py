from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.regimes.regime_registry import (  # noqa: E402
    REGIME_ALLOWED_BASE_FEATURES,
    assert_no_forbidden_regime_features,
    assert_regime_features_are_point_in_time,
    get_allowed_diagnostic_labels,
    get_allowed_regime_features,
    is_regime_feature_available,
)
from vrp.regimes.threshold import (  # noqa: E402
    classify_vrp_zscore,
    rolling_percentile_threshold,
)
from .test_threshold_regimes import _base_config, _synthetic_panel  # noqa: E402


def test_no_forbidden_feature_substrings_in_regime_feature_list():
    forbidden_tokens = ["future", "forward", "expost", "label"]

    for feature in get_allowed_regime_features():
        lower = feature.lower()
        assert not any(token in lower for token in forbidden_tokens)


def test_thresholds_are_shifted_by_one_row():
    series = pd.Series([1.0, 2.0, 3.0, 100.0])

    shifted = rolling_percentile_threshold(
        series,
        window=3,
        percentile=0.80,
        min_periods=3,
        strict_prior=True,
        interpolation="linear",
    )

    unshifted = rolling_percentile_threshold(
        series,
        window=3,
        percentile=0.80,
        min_periods=3,
        strict_prior=False,
        interpolation="linear",
    )

    assert shifted.iloc[-1] == pytest.approx(2.6)
    assert unshifted.iloc[-1] == pytest.approx(61.2)
    assert shifted.iloc[-1] != unshifted.iloc[-1]


def test_full_sample_percentile_is_not_used_for_primary_regime_assignment():
    series = pd.Series([1.0, 2.0, 3.0, 100.0])

    strict_prior = rolling_percentile_threshold(
        series,
        window=3,
        percentile=0.80,
        min_periods=3,
        strict_prior=True,
        interpolation="linear",
    )

    full_sample_threshold = series.quantile(0.80, interpolation="linear")

    assert full_sample_threshold == pytest.approx(41.8)
    assert strict_prior.iloc[-1] == pytest.approx(2.6)
    assert strict_prior.iloc[-1] != full_sample_threshold


@pytest.mark.parametrize(
    "bad_col",
    [
        "rv_gk_22d_forward_ann_label",
        "vrp_forward_expost_gk_label",
        "future_realized_variance",
        "next_day_label",
    ],
)
def test_regime_assignment_rejects_label_expost_forward_columns(bad_col: str):
    with pytest.raises(ValueError):
        assert_no_forbidden_regime_features(["iv_ann", bad_col])

    with pytest.raises(ValueError):
        assert_regime_features_are_point_in_time(["iv_ann", bad_col])


def test_har_based_regime_feature_valid_only_when_har_available_true():
    good_row = {
        "vrp_har_gk": 0.10,
        "har_forecast_available": True,
    }
    bad_row = {
        "vrp_har_gk": 0.10,
        "har_forecast_available": False,
    }
    missing_row = {
        "vrp_har_gk": pd.NA,
        "har_forecast_available": True,
    }

    assert is_regime_feature_available(good_row, "vrp_har_gk") is True
    assert is_regime_feature_available(bad_row, "vrp_har_gk") is False
    assert is_regime_feature_available(missing_row, "vrp_har_gk") is False


def test_no_threshold_state_is_backfilled_for_har_unavailable_row():
    config = _base_config()
    panel = _synthetic_panel(5)

    panel["har_forecast_available"] = [True, True, True, True, False]

    out = classify_vrp_zscore(panel, config)

    assert pd.isna(out.loc[4, "vrp_har_state"])
    assert out.loc[4, "vrp_har_blocked_reason"] == "har_forecast_unavailable"


def test_forward_labels_are_registered_only_as_diagnostics():
    diagnostic_labels = get_allowed_diagnostic_labels()

    assert "rv_gk_22d_forward_ann_label" in diagnostic_labels
    assert "vrp_forward_expost_gk_label" in diagnostic_labels

    for label_col in diagnostic_labels:
        assert label_col not in REGIME_ALLOWED_BASE_FEATURES

        with pytest.raises(ValueError):
            assert_regime_features_are_point_in_time(["iv_ann", label_col])


def test_crisis_windows_do_not_affect_states():
    from vrp.regimes.threshold import classify_all_threshold_components, combine_threshold_regimes

    config_a = _base_config()
    config_b = _base_config()

    config_b["crisis_windows"] = {
        "US": [["1900-01-01", "1900-01-31", "FakeOldCrisis"]],
        "INDIA": [["2100-01-01", "2100-01-31", "FakeFutureCrisis"]],
    }

    panel = _synthetic_panel(8)

    out_a = combine_threshold_regimes(
        classify_all_threshold_components(panel, config_a),
        config_a,
    )

    out_b = combine_threshold_regimes(
        classify_all_threshold_components(panel, config_b),
        config_b,
    )

    assert out_a["threshold_state"].tolist() == out_b["threshold_state"].tolist()
    assert out_a["threshold_trigger_reason"].tolist() == out_b[
        "threshold_trigger_reason"
    ].tolist()


def test_us_india_thresholds_are_not_pooled_by_config_policy():
    config = _base_config()

    assert config["threshold_policy"]["prohibit_cross_market_threshold_pooling"] is True
    assert config["primary_input_files"]["US"] != config["primary_input_files"]["INDIA"]


def test_no_manual_state_overrides_policy_is_true():
    config = _base_config()

    assert config["regime_learning_policy"]["no_manual_state_overrides"] is True
    assert config["regime_learning_policy"]["uses_manual_crisis_labels_for_training"] is False
    assert config["diagnostic_window_policy"]["do_not_tune_thresholds_on_crisis_windows"] is True