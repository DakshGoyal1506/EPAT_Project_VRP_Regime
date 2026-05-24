from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from vrp.backtest.backtest_config import (
    BacktestConfigError,
    get_market_backtest_inputs,
    get_market_output_path,
    get_strategy_universe,
    load_backtest_config,
    validate_backtest_config,
)
from vrp.backtest.backtest_registry import (
    BACKTEST_STRATEGY_UNIVERSE,
    BacktestRegistryError,
    assert_no_msvol_strategy_use,
    assert_no_outcome_labels_used_as_signals,
    assert_no_smoothed_probability_use,
    assert_payoff_label_is_outcome_only,
    assert_strategy_universe_locked,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CONFIG_PATH = REPO_ROOT / "configs" / "backtest.yaml"


def _load_raw_config() -> dict:
    with BACKTEST_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert isinstance(raw, dict)
    return raw


def _write_temp_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "backtest.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_backtest_yaml_loads_and_validates() -> None:
    config = load_backtest_config(BACKTEST_CONFIG_PATH)

    assert config.backtest_phase == "phase_10"
    assert config.primary_payoff.label_col == "vrp_forward_expost_gk_label"
    assert config.primary_payoff.label_role == "realised_outcome_only"
    assert config.primary_payoff.outcome_alignment == "signal_observation_date"
    assert config.primary_payoff.horizon_trading_days == 22
    assert config.primary_payoff.overlapping_labels is True
    assert config.primary_payoff.report_as_research_proxy is True

    validate_backtest_config(config)


def test_strategy_universe_is_locked() -> None:
    assert_strategy_universe_locked(BACKTEST_STRATEGY_UNIVERSE)

    with pytest.raises(BacktestRegistryError):
        assert_strategy_universe_locked(["unconditional_full"])

    with pytest.raises(BacktestRegistryError):
        assert_strategy_universe_locked(
            list(BACKTEST_STRATEGY_UNIVERSE) + ["new_strategy"]
        )


def test_config_strategy_universe_accessor() -> None:
    config = load_backtest_config(BACKTEST_CONFIG_PATH)
    strategies = get_strategy_universe(config)

    assert strategies == BACKTEST_STRATEGY_UNIVERSE


def test_market_input_and_output_accessors() -> None:
    config = load_backtest_config(BACKTEST_CONFIG_PATH)

    us_inputs = get_market_backtest_inputs(config, "US")
    india_inputs = get_market_backtest_inputs(config, "INDIA")

    assert us_inputs["strategy_signals"] == Path("data/processed/us_strategy_signals.parquet")
    assert india_inputs["strategy_signals"] == Path(
        "data/processed/india_strategy_signals.parquet"
    )

    assert get_market_output_path(config, "US") == Path(
        "data/processed/us_backtest_panel.parquet"
    )
    assert get_market_output_path(config, "INDIA", "metadata") == Path(
        "data/processed/india_backtest_panel_metadata.json"
    )


def test_payoff_label_is_outcome_only() -> None:
    assert_payoff_label_is_outcome_only("vrp_forward_expost_gk_label")

    with pytest.raises(BacktestRegistryError):
        assert_payoff_label_is_outcome_only("rv_gk_22d_forward_ann_label")

    with pytest.raises(BacktestRegistryError):
        assert_payoff_label_is_outcome_only("random_label")


def test_forbidden_outcome_labels_cannot_be_signal_columns() -> None:
    with pytest.raises(BacktestRegistryError):
        assert_no_outcome_labels_used_as_signals(
            ["target_exposure", "vrp_forward_expost_gk_label"]
        )

    with pytest.raises(BacktestRegistryError):
        assert_no_outcome_labels_used_as_signals(
            ["rv_gk_22d_forward_ann_label"]
        )

    assert_no_outcome_labels_used_as_signals(
        ["target_exposure", "strategy_available", "hmm_filtered_stress_prob"]
    )


def test_msvol_strategy_is_forbidden() -> None:
    df = pd.DataFrame(
        {
            "strategy_name": [
                "unconditional_full",
                "msvol_appendix_strategy",
            ]
        }
    )

    with pytest.raises(BacktestRegistryError):
        assert_no_msvol_strategy_use(df)


def test_smoothed_probability_columns_are_forbidden() -> None:
    df = pd.DataFrame(
        {
            "hmm_smoothed_stress_prob": [0.1, 0.2],
            "target_exposure": [-1.0, 0.0],
        }
    )

    with pytest.raises(BacktestRegistryError):
        assert_no_smoothed_probability_use(df)


def test_smoothed_probability_source_is_forbidden() -> None:
    df = pd.DataFrame(
        {
            "probability_source": ["filtered", "full_sample_smoothed"],
            "target_exposure": [-1.0, 0.0],
        }
    )

    with pytest.raises(BacktestRegistryError):
        assert_no_smoothed_probability_use(df)


def test_negative_default_cost_fails(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["costs"]["default_cost_bps"] = -1

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestConfigError):
        load_backtest_config(path)


def test_negative_cost_grid_fails(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["robustness"]["cost_bps_grid"] = [0, 5, -10]

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestConfigError):
        load_backtest_config(path)


def test_wrong_horizon_fails_without_override(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["primary_payoff"]["horizon_trading_days"] = 10
    raw["primary_payoff"]["allow_horizon_override"] = False

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestConfigError):
        load_backtest_config(path)


def test_wrong_horizon_passes_with_explicit_override(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["primary_payoff"]["horizon_trading_days"] = 10
    raw["primary_payoff"]["allow_horizon_override"] = True

    path = _write_temp_config(tmp_path, raw)
    config = load_backtest_config(path)

    assert config.primary_payoff.horizon_trading_days == 10
    assert config.primary_payoff.allow_horizon_override is True


def test_payoff_role_must_be_realised_outcome_only(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["primary_payoff"]["label_role"] = "signal_feature"

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestConfigError):
        load_backtest_config(path)


def test_declared_signal_features_cannot_include_payoff_label(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["signal_features"] = ["hmm_filtered_stress_prob", "vrp_forward_expost_gk_label"]

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestRegistryError):
        load_backtest_config(path)


def test_outcome_alignment_must_be_allowed(tmp_path: Path) -> None:
    raw = deepcopy(_load_raw_config())
    raw["primary_payoff"]["outcome_alignment"] = "same_day_return"

    path = _write_temp_config(tmp_path, raw)

    with pytest.raises(BacktestRegistryError):
        load_backtest_config(path)