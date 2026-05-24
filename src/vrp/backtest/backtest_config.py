from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vrp.backtest.backtest_registry import (
    BACKTEST_FORBIDDEN_SIGNAL_COLUMNS,
    assert_no_outcome_labels_used_as_signals,
    assert_outcome_alignment_allowed,
    assert_payoff_label_is_outcome_only,
    assert_strategy_universe_locked,
)


PHASE_10_EXPECTED_HORIZON_TRADING_DAYS = 22
SUPPORTED_MARKETS: tuple[str, ...] = ("US", "INDIA")


class BacktestConfigError(ValueError):
    """Raised when configs/backtest.yaml is invalid."""


@dataclass(frozen=True)
class PayoffConfig:
    name: str
    label_col: str
    label_role: str
    outcome_alignment: str
    payoff_formula: str
    annualization_periods: int
    horizon_trading_days: int
    allow_horizon_override: bool
    overlapping_labels: bool
    report_as_research_proxy: bool


@dataclass(frozen=True)
class CostConfig:
    enabled: bool
    default_cost_bps: float
    apply_to_abs_exposure_change: bool
    cost_formula: str


@dataclass(frozen=True)
class RobustnessConfig:
    cost_bps_grid: tuple[float, ...]
    rebalance_frequencies: tuple[str, ...]
    subperiods: dict[str, tuple[tuple[str, str, str], ...]]


@dataclass(frozen=True)
class BacktestConfig:
    backtest_phase: str
    description: str
    input_files: dict[str, dict[str, str]]
    strategy_universe: tuple[str, ...]
    primary_payoff: PayoffConfig
    costs: CostConfig
    robustness: RobustnessConfig
    output_files: dict[str, dict[str, str]]
    reporting: dict[str, str]
    raw: dict[str, Any] = field(repr=False, compare=False)


def _require_mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BacktestConfigError(f"{name} must be a mapping.")
    return raw


def _require_keys(raw: dict[str, Any], required_keys: list[str], name: str) -> None:
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise BacktestConfigError(f"{name} is missing required keys: {missing}")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise BacktestConfigError(f"{name} must be a boolean.")


def _as_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise BacktestConfigError(f"{name} must be an integer.") from exc

    if parsed <= 0:
        raise BacktestConfigError(f"{name} must be positive. Got {parsed}.")

    return parsed


def _as_non_negative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except Exception as exc:
        raise BacktestConfigError(f"{name} must be numeric.") from exc

    if parsed < 0:
        raise BacktestConfigError(f"{name} must be non-negative. Got {parsed}.")

    return parsed


def _parse_payoff_config(raw: dict[str, Any]) -> PayoffConfig:
    required = [
        "name",
        "label_col",
        "label_role",
        "outcome_alignment",
        "payoff_formula",
        "annualization_periods",
        "horizon_trading_days",
        "overlapping_labels",
        "report_as_research_proxy",
    ]
    _require_keys(raw, required, "primary_payoff")

    return PayoffConfig(
        name=str(raw["name"]),
        label_col=str(raw["label_col"]),
        label_role=str(raw["label_role"]),
        outcome_alignment=str(raw["outcome_alignment"]),
        payoff_formula=str(raw["payoff_formula"]),
        annualization_periods=_as_positive_int(
            raw["annualization_periods"],
            "primary_payoff.annualization_periods",
        ),
        horizon_trading_days=_as_positive_int(
            raw["horizon_trading_days"],
            "primary_payoff.horizon_trading_days",
        ),
        allow_horizon_override=bool(raw.get("allow_horizon_override", False)),
        overlapping_labels=_as_bool(
            raw["overlapping_labels"],
            "primary_payoff.overlapping_labels",
        ),
        report_as_research_proxy=_as_bool(
            raw["report_as_research_proxy"],
            "primary_payoff.report_as_research_proxy",
        ),
    )


def _parse_cost_config(raw: dict[str, Any]) -> CostConfig:
    required = [
        "enabled",
        "default_cost_bps",
        "apply_to_abs_exposure_change",
        "cost_formula",
    ]
    _require_keys(raw, required, "costs")

    return CostConfig(
        enabled=_as_bool(raw["enabled"], "costs.enabled"),
        default_cost_bps=_as_non_negative_float(
            raw["default_cost_bps"],
            "costs.default_cost_bps",
        ),
        apply_to_abs_exposure_change=_as_bool(
            raw["apply_to_abs_exposure_change"],
            "costs.apply_to_abs_exposure_change",
        ),
        cost_formula=str(raw["cost_formula"]),
    )


def _parse_robustness_config(raw: dict[str, Any]) -> RobustnessConfig:
    required = [
        "cost_bps_grid",
        "rebalance_frequencies",
        "subperiods",
    ]
    _require_keys(raw, required, "robustness")

    cost_grid_raw = raw["cost_bps_grid"]
    if not isinstance(cost_grid_raw, list):
        raise BacktestConfigError("robustness.cost_bps_grid must be a list.")

    cost_grid = tuple(
        _as_non_negative_float(value, "robustness.cost_bps_grid")
        for value in cost_grid_raw
    )

    rebalance_raw = raw["rebalance_frequencies"]
    if not isinstance(rebalance_raw, list):
        raise BacktestConfigError("robustness.rebalance_frequencies must be a list.")

    rebalance_frequencies = tuple(str(value) for value in rebalance_raw)

    subperiods_raw = _require_mapping(raw["subperiods"], "robustness.subperiods")
    subperiods: dict[str, tuple[tuple[str, str, str], ...]] = {}

    for market, rows in subperiods_raw.items():
        market_key = str(market).upper()
        if not isinstance(rows, list):
            raise BacktestConfigError(
                f"robustness.subperiods.{market_key} must be a list."
            )

        parsed_rows: list[tuple[str, str, str]] = []
        for row in rows:
            if not isinstance(row, list | tuple) or len(row) != 3:
                raise BacktestConfigError(
                    f"Each robustness subperiod for {market_key} must be "
                    "[start_date, end_date, label]."
                )
            parsed_rows.append((str(row[0]), str(row[1]), str(row[2])))

        subperiods[market_key] = tuple(parsed_rows)

    return RobustnessConfig(
        cost_bps_grid=cost_grid,
        rebalance_frequencies=rebalance_frequencies,
        subperiods=subperiods,
    )


def _parse_config(raw: dict[str, Any]) -> BacktestConfig:
    required = [
        "backtest_phase",
        "description",
        "input_files",
        "strategy_universe",
        "primary_payoff",
        "costs",
        "robustness",
        "output_files",
        "reporting",
    ]
    _require_keys(raw, required, "backtest config")

    input_files = _require_mapping(raw["input_files"], "input_files")
    output_files = _require_mapping(raw["output_files"], "output_files")
    reporting = _require_mapping(raw["reporting"], "reporting")

    strategy_universe_raw = raw["strategy_universe"]
    if not isinstance(strategy_universe_raw, list):
        raise BacktestConfigError("strategy_universe must be a list.")

    config = BacktestConfig(
        backtest_phase=str(raw["backtest_phase"]),
        description=str(raw["description"]),
        input_files={
            str(market).upper(): {
                str(key): str(value)
                for key, value in _require_mapping(paths, f"input_files.{market}").items()
            }
            for market, paths in input_files.items()
        },
        strategy_universe=tuple(str(value) for value in strategy_universe_raw),
        primary_payoff=_parse_payoff_config(
            _require_mapping(raw["primary_payoff"], "primary_payoff")
        ),
        costs=_parse_cost_config(_require_mapping(raw["costs"], "costs")),
        robustness=_parse_robustness_config(
            _require_mapping(raw["robustness"], "robustness")
        ),
        output_files={
            str(market).upper(): {
                str(key): str(value)
                for key, value in _require_mapping(paths, f"output_files.{market}").items()
            }
            for market, paths in output_files.items()
        },
        reporting={str(key): str(value) for key, value in reporting.items()},
        raw=raw,
    )

    return config


def _collect_declared_signal_columns(raw: dict[str, Any]) -> list[str]:
    candidate_keys = {
        "signal_columns",
        "signal_cols",
        "signal_features",
        "feature_cols",
        "features_for_signal",
    }

    found: list[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key) in candidate_keys:
                    if isinstance(value, list):
                        found.extend(str(item) for item in value)
                    elif isinstance(value, str):
                        found.append(value)
                    else:
                        raise BacktestConfigError(
                            f"Signal-column declaration {key!r} must be string or list."
                        )
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(raw)
    return found


def validate_backtest_config(config: BacktestConfig) -> None:
    if config.backtest_phase != "phase_10":
        raise BacktestConfigError(
            f"backtest_phase must be 'phase_10'. Got {config.backtest_phase!r}."
        )

    assert_strategy_universe_locked(config.strategy_universe)

    payoff = config.primary_payoff

    assert_payoff_label_is_outcome_only(payoff.label_col)
    assert_outcome_alignment_allowed(payoff.outcome_alignment)

    if payoff.label_role != "realised_outcome_only":
        raise BacktestConfigError(
            "primary_payoff.label_role must be 'realised_outcome_only'. "
            f"Got {payoff.label_role!r}."
        )

    declared_signal_cols = _collect_declared_signal_columns(config.raw)
    assert_no_outcome_labels_used_as_signals(declared_signal_cols)

    if payoff.label_col not in BACKTEST_FORBIDDEN_SIGNAL_COLUMNS:
        raise BacktestConfigError(
            f"primary_payoff.label_col={payoff.label_col!r} must be forbidden "
            "as a signal column."
        )

    formula_compact = payoff.payoff_formula.replace(" ", "")
    if formula_compact != "-target_exposure*label":
        raise BacktestConfigError(
            "primary_payoff.payoff_formula must be '-target_exposure * label'. "
            f"Got {payoff.payoff_formula!r}."
        )

    if not payoff.allow_horizon_override:
        if payoff.horizon_trading_days != PHASE_10_EXPECTED_HORIZON_TRADING_DAYS:
            raise BacktestConfigError(
                "primary_payoff.horizon_trading_days must match the Phase 3/4 "
                f"default horizon of {PHASE_10_EXPECTED_HORIZON_TRADING_DAYS}, "
                "unless allow_horizon_override is true. "
                f"Got {payoff.horizon_trading_days}."
            )

    if not payoff.overlapping_labels:
        raise BacktestConfigError(
            "primary_payoff.overlapping_labels must be true for the default "
            "22-trading-day forward VRP labels."
        )

    if not payoff.report_as_research_proxy:
        raise BacktestConfigError(
            "primary_payoff.report_as_research_proxy must be true."
        )

    if config.costs.default_cost_bps < 0:
        raise BacktestConfigError("costs.default_cost_bps must be non-negative.")

    if config.costs.enabled and not config.costs.apply_to_abs_exposure_change:
        raise BacktestConfigError(
            "Phase 10 cost model must apply to absolute exposure change."
        )

    allowed_rebalance = {"daily", "weekly"}
    invalid_rebalance = [
        freq for freq in config.robustness.rebalance_frequencies
        if freq not in allowed_rebalance
    ]
    if invalid_rebalance:
        raise BacktestConfigError(
            f"Unsupported rebalance frequencies: {invalid_rebalance}. "
            f"Allowed: {sorted(allowed_rebalance)}"
        )

    if "daily" not in config.robustness.rebalance_frequencies:
        raise BacktestConfigError(
            "robustness.rebalance_frequencies must include 'daily'."
        )

    for market in SUPPORTED_MARKETS:
        if market not in config.input_files:
            raise BacktestConfigError(f"input_files missing market {market}.")
        if market not in config.output_files:
            raise BacktestConfigError(f"output_files missing market {market}.")

        input_required = ["strategy_signals", "vrp_har", "vrp", "threshold", "hmm", "mar"]
        output_required = ["backtest_panel", "metadata"]

        missing_input = [
            key for key in input_required
            if key not in config.input_files[market]
        ]
        missing_output = [
            key for key in output_required
            if key not in config.output_files[market]
        ]

        if missing_input:
            raise BacktestConfigError(
                f"input_files.{market} missing required keys: {missing_input}"
            )

        if missing_output:
            raise BacktestConfigError(
                f"output_files.{market} missing required keys: {missing_output}"
            )

    for market in config.robustness.subperiods:
        if market not in SUPPORTED_MARKETS:
            raise BacktestConfigError(
                f"robustness.subperiods contains unsupported market {market!r}."
            )

    if "table_dir" not in config.reporting:
        raise BacktestConfigError("reporting.table_dir is required.")

    if "figure_dir" not in config.reporting:
        raise BacktestConfigError("reporting.figure_dir is required.")


def load_backtest_config(path: str | Path) -> BacktestConfig:
    path = Path(path)

    if not path.exists():
        raise BacktestConfigError(f"Backtest config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    raw = _require_mapping(raw, "backtest config")
    config = _parse_config(raw)
    validate_backtest_config(config)

    return config


def _normalize_market(market: str) -> str:
    market_key = market.upper()
    if market_key not in SUPPORTED_MARKETS:
        raise BacktestConfigError(
            f"Unsupported market {market!r}. Expected one of {SUPPORTED_MARKETS}."
        )
    return market_key


def get_market_backtest_inputs(
    config: BacktestConfig,
    market: str,
) -> dict[str, Path]:
    market_key = _normalize_market(market)
    return {
        key: Path(value)
        for key, value in config.input_files[market_key].items()
    }


def get_strategy_universe(config: BacktestConfig) -> tuple[str, ...]:
    assert_strategy_universe_locked(config.strategy_universe)
    return config.strategy_universe


def get_market_output_path(
    config: BacktestConfig,
    market: str,
    output_key: str = "backtest_panel",
) -> Path:
    market_key = _normalize_market(market)

    if output_key not in config.output_files[market_key]:
        raise BacktestConfigError(
            f"output_files.{market_key} does not contain {output_key!r}."
        )

    return Path(config.output_files[market_key][output_key])