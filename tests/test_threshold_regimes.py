from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.regimes.state_labeling import (  # noqa: E402
    CALM,
    STRESS,
    TRANSITION,
    STATE_ID_TO_NAME,
    STATE_NAME_TO_ID,
    map_state_id_to_name,
    map_state_name_to_id,
    validate_state_mapping_consistency,
)
from vrp.regimes.threshold import (  # noqa: E402
    classify_all_threshold_components,
    classify_drawdown,
    classify_iv_percentile,
    classify_iv_slope,
    classify_rv_percentile,
    classify_vrp_zscore,
    combine_threshold_regimes,
    expanding_percentile_threshold,
    rolling_percentile_threshold,
    validate_threshold_config,
)


def _base_config() -> dict:
    return {
        "model_name": "threshold_baseline_v1",
        "primary_input_files": {
            "US": "data/processed/us_vrp_har.parquet",
            "INDIA": "data/processed/india_vrp_har.parquet",
        },
        "primary_output_files": {
            "US": "data/processed/us_threshold_regimes.parquet",
            "INDIA": "data/processed/india_threshold_regimes.parquet",
        },
        "annualization_periods": 252,
        "states": {
            "calm": 0,
            "transition": 1,
            "stress": 2,
        },
        "regime_learning_policy": {
            "regime_type": "deterministic_unsupervised_threshold",
            "uses_manual_crisis_labels_for_training": False,
            "uses_crisis_windows_for_reporting_only": True,
            "uses_forward_labels_for_training": False,
            "uses_forward_labels_for_reporting_only": True,
            "no_manual_state_overrides": True,
        },
        "diagnostic_window_policy": {
            "use_crisis_windows_only_for_reporting": True,
            "skip_windows_outside_sample": True,
            "do_not_tune_thresholds_on_crisis_windows": True,
        },
        "threshold_policy": {
            "method": "rolling_expanding",
            "use_strict_prior_thresholds": True,
            "min_history": 3,
            "rolling_window": 3,
            "percentile_interpolation": "linear",
            "prohibit_full_sample_percentiles": True,
            "prohibit_cross_market_threshold_pooling": True,
        },
        "iv_filter": {
            "feature_col": "iv_ann",
            "calm_percentile": 0.50,
            "stress_percentile": 0.80,
            "high_is_stress": True,
            "required": True,
        },
        "rv_filter": {
            "feature_col": "rv_gk_22d_ann_lag1",
            "calm_percentile": 0.50,
            "stress_percentile": 0.80,
            "high_is_stress": True,
            "required": True,
        },
        "drawdown_filter": {
            "price_col_candidates": ["close", "adj_close", "underlying_close"],
            "fallback_return_col_candidates": ["log_return", "simple_return"],
            "lookback_window": 3,
            "transition_drawdown": -0.07,
            "stress_drawdown": -0.12,
            "required": False,
        },
        "iv_slope_filter": {
            "feature_col": "iv_ann",
            "short_ma_window": 2,
            "long_ma_window": 4,
            "transition_slope_threshold": 0.00,
            "stress_slope_threshold": 0.10,
            "required": False,
        },
        "vrp_har_filter": {
            "feature_col": "vrp_har_gk",
            "require_har_forecast_available": True,
            "availability_col": "har_forecast_available",
            "zscore_window": 3,
            "min_history": 3,
            "calm_zscore": 0.00,
            "stress_zscore": -1.00,
            "required": True,
        },
        "component_policy": {
            "required_components": [
                "iv_percentile_state",
                "rv_percentile_state",
                "vrp_har_state",
            ],
            "optional_components": [
                "drawdown_state",
                "iv_slope_state",
            ],
            "do_not_fill_missing_component_states": True,
            "do_not_backfill_component_states": True,
            "do_not_forward_fill_component_states": True,
            "missing_required_blocks_final_regime": True,
            "score_optional_components_when_available_only": True,
            "score_missing_components_as": None,
        },
        "combined_filter": {
            "score_method": "weighted_sum",
            "component_weights": {
                "iv_percentile_state": 1.0,
                "rv_percentile_state": 1.0,
                "drawdown_state": 1.5,
                "iv_slope_state": 0.5,
                "vrp_har_state": 1.0,
            },
            "calm_score_cutoff": 1.0,
            "stress_score_cutoff": 2.5,
            "hard_stress_components": [
                "iv_percentile_state",
                "rv_percentile_state",
                "drawdown_state",
            ],
            "calm_required_conditions": {
                "iv_percentile_state": "calm",
                "rv_percentile_state_not": "stress",
                "drawdown_state": "calm_or_unavailable",
                "vrp_har_state": "calm",
            },
            "require_score_cutoffs_reachable": True,
            "write_trigger_explanations": True,
        },
        "sample_coverage_policy": {
            "warn_if_available_fraction_after_warmup_below": 0.60,
            "hard_fail_if_no_available_regimes": True,
        },
        "metadata_policy": {
            "write_config_sha256": True,
            "write_input_file_metadata": True,
            "write_git_commit_if_available": True,
            "write_created_at_utc": True,
        },
        "diagnostics": {
            "write_reports": True,
            "write_figures": True,
            "allow_forward_labels_only_after_regime_assignment": True,
            "duration_diagnostics": True,
            "annual_state_distribution": True,
            "crisis_hit_table": True,
            "crisis_lead_lag_table": True,
            "forward_label_by_state_table": True,
            "no_lookahead_audit": True,
        },
        "crisis_windows": {
            "US": [["2020-02-20", "2020-06-30", "COVID"]],
            "INDIA": [["2020-02-20", "2020-06-30", "COVID"]],
        },
    }


def _synthetic_panel(n: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "date": dates,
            "market": ["US"] * n,
            "iv_ann": np.linspace(0.10, 0.20, n),
            "iv_close": np.linspace(10.0, 20.0, n),
            "rv_gk_22d_ann_lag1": np.linspace(0.08, 0.18, n),
            "vrp_backward_gk": np.linspace(0.02, 0.03, n),
            "vrp_backward_gk_positive": [True] * n,
            "har_rv_gk_22d_forecast_ann": np.linspace(0.09, 0.19, n),
            "vrp_har_gk": [1.00, 1.10, 0.90, -3.00, 1.00, 1.10, 0.90, 1.20][:n],
            "vrp_har_gk_positive": [True, True, True, False, True, True, True, True][:n],
            "har_forecast_available": [True] * n,
            "close": [100, 102, 101, 99, 98, 97, 96, 95][:n],
            "log_return": [0.0] * n,
            "simple_return": [0.0] * n,
            "feature_allowed": [True] * n,
            "rv_gk_22d_forward_ann_label": np.linspace(0.10, 0.22, n),
            "vrp_forward_expost_gk_label": np.linspace(0.02, -0.02, n),
        }
    )


def test_state_label_mapping_is_canonical_and_reversible():
    assert CALM == 0
    assert TRANSITION == 1
    assert STRESS == 2

    assert STATE_ID_TO_NAME == {
        0: "calm",
        1: "transition",
        2: "stress",
    }

    assert STATE_NAME_TO_ID == {
        "calm": 0,
        "transition": 1,
        "stress": 2,
    }

    assert validate_state_mapping_consistency() is True

    ids = pd.Series([CALM, TRANSITION, STRESS, pd.NA], dtype="Int64")
    names = map_state_id_to_name(ids)

    assert names.tolist()[:3] == ["calm", "transition", "stress"]

    roundtrip = map_state_name_to_id(names)
    assert roundtrip.tolist()[:3] == [CALM, TRANSITION, STRESS]


def test_rolling_threshold_uses_strict_prior_values():
    series = pd.Series([1.0, 2.0, 3.0, 100.0])

    threshold = rolling_percentile_threshold(
        series,
        window=3,
        percentile=0.80,
        min_periods=3,
        strict_prior=True,
        interpolation="linear",
    )

    assert threshold.iloc[-1] == pytest.approx(2.6)
    assert threshold.iloc[-1] < 100.0


def test_expanding_threshold_uses_strict_prior_values():
    series = pd.Series([1.0, 2.0, 3.0, 100.0])

    threshold = expanding_percentile_threshold(
        series,
        percentile=0.80,
        min_periods=3,
        strict_prior=True,
        interpolation="linear",
    )

    assert threshold.iloc[-1] == pytest.approx(2.6)
    assert threshold.iloc[-1] < 100.0


def test_iv_percentile_high_iv_classifies_as_stress():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["iv_ann"] = [1.0, 2.0, 3.0, 100.0]

    out = classify_iv_percentile(panel, config)

    assert out.loc[3, "iv_percentile_state"] == STRESS
    assert out.loc[3, "iv_percentile_state_name"] == "stress"
    assert out.loc[3, "iv_percentile_available"] is True or bool(
        out.loc[3, "iv_percentile_available"]
    ) is True


def test_rv_percentile_high_rv_classifies_as_stress():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["rv_gk_22d_ann_lag1"] = [1.0, 2.0, 3.0, 100.0]

    out = classify_rv_percentile(panel, config)

    assert out.loc[3, "rv_percentile_state"] == STRESS
    assert out.loc[3, "rv_percentile_state_name"] == "stress"


def test_drawdown_below_stress_threshold_classifies_as_stress():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["close"] = [100.0, 120.0, 110.0, 90.0]

    out = classify_drawdown(panel, config)

    assert out.loc[3, "drawdown"] == pytest.approx(-0.25)
    assert out.loc[3, "drawdown_state"] == STRESS
    assert out.loc[3, "drawdown_state_name"] == "stress"


def test_iv_slope_rising_short_ma_classifies_as_stress():
    config = _base_config()
    panel = _synthetic_panel(5)
    panel["iv_ann"] = [1.0, 1.0, 3.0, 3.0, 4.0]

    out = classify_iv_slope(panel, config)

    assert out.loc[4, "iv_slope"] == pytest.approx(0.5)
    assert out.loc[4, "iv_slope_state"] == STRESS


def test_vrp_har_very_negative_zscore_classifies_as_stress():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["vrp_har_gk"] = [1.0, 1.1, 0.9, -3.0]
    panel["har_forecast_available"] = [True, True, True, True]

    out = classify_vrp_zscore(panel, config)

    assert float(out.loc[3, "vrp_har_zscore"]) < -1.0
    assert out.loc[3, "vrp_har_state"] == STRESS


def test_har_unavailable_rows_do_not_receive_vrp_har_state():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["vrp_har_gk"] = [1.0, 1.1, 0.9, -3.0]
    panel["har_forecast_available"] = [True, True, True, False]

    out = classify_vrp_zscore(panel, config)

    assert pd.isna(out.loc[3, "vrp_har_state"])
    assert out.loc[3, "vrp_har_available"] is False or bool(
        out.loc[3, "vrp_har_available"]
    ) is False
    assert out.loc[3, "vrp_har_blocked_reason"] == "har_forecast_unavailable"


def test_combined_regime_hard_stress_forces_final_stress():
    config = _base_config()

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1),
            "market": ["US"],
            "iv_percentile_state": [STRESS],
            "rv_percentile_state": [CALM],
            "drawdown_state": [CALM],
            "iv_slope_state": [CALM],
            "vrp_har_state": [CALM],
            "iv_percentile_available": [True],
            "rv_percentile_available": [True],
            "drawdown_available": [True],
            "iv_slope_available": [True],
            "vrp_har_available": [True],
            "iv_percentile_blocked_reason": [pd.NA],
            "rv_percentile_blocked_reason": [pd.NA],
            "drawdown_blocked_reason": [pd.NA],
            "iv_slope_blocked_reason": [pd.NA],
            "vrp_har_blocked_reason": [pd.NA],
            "iv_percentile_state_name": ["stress"],
            "rv_percentile_state_name": ["calm"],
            "drawdown_state_name": ["calm"],
            "iv_slope_state_name": ["calm"],
            "vrp_har_state_name": ["calm"],
            "iv_ann": [1.0],
            "iv_close": [10.0],
            "rv_gk_22d_ann_lag1": [1.0],
            "vrp_backward_gk": [0.1],
            "har_rv_gk_22d_forecast_ann": [1.0],
            "vrp_har_gk": [0.1],
            "har_forecast_available": [True],
        }
    )

    out = combine_threshold_regimes(panel, config)

    assert out.loc[0, "threshold_state"] == STRESS
    assert out.loc[0, "threshold_state_name"] == "stress"
    assert out.loc[0, "threshold_trigger_reason"] == "hard_stress:iv_percentile_state"


def test_missing_required_component_blocks_final_regime():
    config = _base_config()

    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=1),
            "market": ["US"],
            "iv_percentile_state": [CALM],
            "rv_percentile_state": [CALM],
            "drawdown_state": [CALM],
            "iv_slope_state": [CALM],
            "vrp_har_state": [pd.NA],
            "iv_percentile_available": [True],
            "rv_percentile_available": [True],
            "drawdown_available": [True],
            "iv_slope_available": [True],
            "vrp_har_available": [False],
            "iv_percentile_blocked_reason": [pd.NA],
            "rv_percentile_blocked_reason": [pd.NA],
            "drawdown_blocked_reason": [pd.NA],
            "iv_slope_blocked_reason": [pd.NA],
            "vrp_har_blocked_reason": ["har_forecast_unavailable"],
            "iv_percentile_state_name": ["calm"],
            "rv_percentile_state_name": ["calm"],
            "drawdown_state_name": ["calm"],
            "iv_slope_state_name": ["calm"],
            "vrp_har_state_name": [pd.NA],
            "iv_ann": [1.0],
            "iv_close": [10.0],
            "rv_gk_22d_ann_lag1": [1.0],
            "vrp_backward_gk": [0.1],
            "har_rv_gk_22d_forecast_ann": [1.0],
            "vrp_har_gk": [np.nan],
            "har_forecast_available": [False],
        }
    )

    out = combine_threshold_regimes(panel, config)

    assert bool(out.loc[0, "threshold_regime_available"]) is False
    assert pd.isna(out.loc[0, "threshold_state"])
    assert "missing_required:vrp_har_state" in str(out.loc[0, "threshold_blocked_reason"])


def test_output_schema_contains_required_columns():
    config = _base_config()
    panel = _synthetic_panel(8)

    components = classify_all_threshold_components(panel, config)
    out = combine_threshold_regimes(components, config)

    required_cols = {
        "date",
        "market",
        "threshold_model_name",
        "iv_percentile_state",
        "iv_percentile_state_name",
        "rv_percentile_state",
        "rv_percentile_state_name",
        "drawdown_state",
        "drawdown_state_name",
        "iv_slope_state",
        "iv_slope_state_name",
        "vrp_har_state",
        "vrp_har_state_name",
        "iv_percentile_available",
        "rv_percentile_available",
        "drawdown_available",
        "iv_slope_available",
        "vrp_har_available",
        "iv_percentile_blocked_reason",
        "rv_percentile_blocked_reason",
        "drawdown_blocked_reason",
        "iv_slope_blocked_reason",
        "vrp_har_blocked_reason",
        "threshold_stress_score",
        "threshold_state",
        "threshold_state_name",
        "threshold_regime_available",
        "threshold_blocked_reason",
        "threshold_hard_stress_trigger",
        "threshold_score_trigger",
        "threshold_calm_conditions_pass",
        "threshold_primary_components_available",
        "threshold_trigger_reason",
        "iv_ann",
        "iv_close",
        "rv_gk_22d_ann_lag1",
        "vrp_backward_gk",
        "har_rv_gk_22d_forecast_ann",
        "vrp_har_gk",
        "har_forecast_available",
        "iv_calm_threshold",
        "iv_stress_threshold",
        "rv_calm_threshold",
        "rv_stress_threshold",
        "vrp_har_zscore",
        "drawdown",
        "iv_slope",
        "threshold_history_start_date",
        "threshold_history_end_date",
        "threshold_n_history",
    }

    assert required_cols.issubset(set(out.columns))


def test_score_cutoff_is_reachable_under_weighted_sum():
    config = _base_config()

    assert validate_threshold_config(config) is True

    weights = config["combined_filter"]["component_weights"]
    max_possible_score = sum(weight * STRESS for weight in weights.values())

    assert config["combined_filter"]["stress_score_cutoff"] <= max_possible_score


def test_unreachable_score_cutoff_is_rejected():
    config = _base_config()
    config["combined_filter"]["stress_score_cutoff"] = 100.0

    with pytest.raises(ValueError, match="unreachable"):
        validate_threshold_config(config)


def test_component_missing_is_not_converted_to_transition():
    config = _base_config()
    panel = _synthetic_panel(4)
    panel["har_forecast_available"] = [True, True, True, False]

    out = classify_vrp_zscore(panel, config)

    # When component is missing, state must be NA (not TRANSITION or any other value)
    assert pd.isna(out.loc[3, "vrp_har_state"])