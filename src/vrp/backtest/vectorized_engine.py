from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from vrp.backtest.backtest_config import (
    BacktestConfig,
    BacktestConfigError,
    get_market_output_path,
)
from vrp.backtest.backtest_registry import (
    BACKTEST_STRATEGY_UNIVERSE,
    assert_strategy_universe_locked,
)
from vrp.backtest.payoff_proxies import build_research_backtest_panel


SUPPORTED_MARKETS: tuple[str, ...] = ("US", "INDIA")


class VectorizedBacktestError(ValueError):
    """Raised when Phase 10 vectorised backtest execution fails."""


@dataclass(frozen=True)
class BacktestRunResult:
    market: str
    strategy: str
    output_path: Path
    metadata_path: Path
    n_rows: int
    n_eligible: int
    n_strategies: int
    cost_bps: float
    wrote_files: bool
    metadata: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if not np.isfinite(float(value)):
            return None
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    return value


def _resolve_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_config_paths(
    config: BacktestConfig,
    *,
    repo_root: Path,
) -> BacktestConfig:
    repo_root = Path(repo_root).resolve()

    input_files = {
        market: {
            key: str(_resolve_path(repo_root, value))
            for key, value in paths.items()
        }
        for market, paths in config.input_files.items()
    }

    output_files = {
        market: {
            key: str(_resolve_path(repo_root, value))
            for key, value in paths.items()
        }
        for market, paths in config.output_files.items()
    }

    reporting = {
        key: str(_resolve_path(repo_root, value))
        for key, value in config.reporting.items()
    }

    return replace(
        config,
        input_files=input_files,
        output_files=output_files,
        reporting=reporting,
    )


def resolve_markets(market: str) -> list[str]:
    market = market.upper()

    if market == "ALL":
        return list(SUPPORTED_MARKETS)

    if market not in SUPPORTED_MARKETS:
        raise VectorizedBacktestError(
            f"Unsupported market {market!r}. Expected US, INDIA, or ALL."
        )

    return [market]


def _normalize_strategy(strategy: str) -> str:
    strategy = str(strategy)

    if strategy == "all":
        return strategy

    if strategy not in BACKTEST_STRATEGY_UNIVERSE:
        raise VectorizedBacktestError(
            f"Unsupported strategy {strategy!r}. Expected 'all' or one of "
            f"{list(BACKTEST_STRATEGY_UNIVERSE)}."
        )

    return strategy


def _ensure_output_can_be_written(
    output_path: Path,
    metadata_path: Path,
    *,
    force: bool,
) -> None:
    existing = [path for path in (output_path, metadata_path) if path.exists()]

    if existing and not force:
        raise VectorizedBacktestError(
            "Output files already exist. Use --force to overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _filter_strategy(panel: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy = _normalize_strategy(strategy)

    if strategy == "all":
        return panel.copy()

    return panel.loc[panel["strategy_name"].astype(str).eq(strategy)].copy()


def _require_panel_columns(panel: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise VectorizedBacktestError(
            f"Backtest panel missing required columns: {missing}"
        )


def validate_backtest_panel_integrity(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> None:
    required = (
        "market",
        "strategy_name",
        "signal_observation_date",
        "target_trade_date",
        "outcome_label_date",
        "target_exposure",
        "target_exposure_for_backtest",
        "is_backtest_eligible",
        "exclusion_reason",
        "gross_return_proxy",
        "cost_proxy",
        "net_return_proxy",
    )
    _require_panel_columns(panel, required)

    strategies = sorted(panel["strategy_name"].dropna().astype(str).unique().tolist())
    invalid = sorted(set(strategies) - set(BACKTEST_STRATEGY_UNIVERSE))
    if invalid:
        raise VectorizedBacktestError(f"Invalid strategies in panel: {invalid}")

    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    if eligible.any():
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

        if signal_dates.isna().any():
            raise VectorizedBacktestError("Eligible rows contain invalid signal dates.")

        if trade_dates.isna().any():
            raise VectorizedBacktestError("Eligible rows contain invalid target trade dates.")

        if outcome_dates.isna().any():
            raise VectorizedBacktestError("Eligible rows contain invalid outcome dates.")

        same_day_or_backward = trade_dates <= signal_dates
        if bool(same_day_or_backward.any()):
            raise VectorizedBacktestError(
                "No-lookahead violation: eligible rows must have "
                "target_trade_date > signal_observation_date."
            )

        if config.primary_payoff.outcome_alignment == "signal_observation_date":
            bad_alignment = outcome_dates != signal_dates
            if bool(bad_alignment.any()):
                raise VectorizedBacktestError(
                    "Outcome alignment violation: outcome_label_date must equal "
                    "signal_observation_date under default Phase 10 alignment."
                )

        exposure = pd.to_numeric(
            panel.loc[eligible, "target_exposure_for_backtest"],
            errors="coerce",
        )
        if exposure.isna().any():
            raise VectorizedBacktestError(
                "Eligible rows contain missing target_exposure_for_backtest."
            )

        net_returns = pd.to_numeric(
            panel.loc[eligible, "net_return_proxy"],
            errors="coerce",
        )
        if net_returns.isna().any():
            raise VectorizedBacktestError(
                "Eligible rows contain missing net_return_proxy."
            )

    allowed_exclusion_reasons = {
        "available",
        "strategy_unavailable",
        "missing_target_trade_date",
        "non_finite_exposure",
        "missing_payoff_label",
        "missing_outcome_join",
        "invalid_strategy_name",
    }
    observed_reasons = set(panel["exclusion_reason"].dropna().astype(str).unique())
    unknown_reasons = sorted(observed_reasons - allowed_exclusion_reasons)
    if unknown_reasons:
        raise VectorizedBacktestError(
            f"Unknown exclusion_reason values: {unknown_reasons}"
        )


def _count_no_lookahead_violations(panel: pd.DataFrame) -> dict[str, int]:
    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    if not eligible.any():
        return {
            "n_target_not_after_signal_violations": 0,
            "n_outcome_not_equal_signal_date_violations": 0,
        }

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

    return {
        "n_target_not_after_signal_violations": int((trade_dates <= signal_dates).sum()),
        "n_outcome_not_equal_signal_date_violations": int((outcome_dates != signal_dates).sum()),
    }


def build_backtest_metadata(
    panel: pd.DataFrame,
    *,
    market: str,
    strategy: str,
    config: BacktestConfig,
    cost_bps: float,
    output_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    strategy_counts = (
        panel.groupby("strategy_name", dropna=False)
        .size()
        .sort_index()
        .to_dict()
    )
    eligible_counts = (
        panel.loc[eligible]
        .groupby("strategy_name", dropna=False)
        .size()
        .sort_index()
        .to_dict()
    )

    exclusion_counts = (
        panel.groupby("exclusion_reason", dropna=False)
        .size()
        .sort_index()
        .to_dict()
    )

    if len(panel) > 0:
        signal_min = pd.to_datetime(panel["signal_observation_date"], errors="coerce").min()
        signal_max = pd.to_datetime(panel["signal_observation_date"], errors="coerce").max()
        trade_min = pd.to_datetime(panel["target_trade_date"], errors="coerce").min()
        trade_max = pd.to_datetime(panel["target_trade_date"], errors="coerce").max()
    else:
        signal_min = signal_max = trade_min = trade_max = None

    metadata = {
        "phase": "phase_10",
        "market": market.upper(),
        "strategy_argument": strategy,
        "payoff_type": config.primary_payoff.name,
        "payoff_label": config.primary_payoff.label_col,
        "label_role": config.primary_payoff.label_role,
        "outcome_alignment": config.primary_payoff.outcome_alignment,
        "payoff_formula": config.primary_payoff.payoff_formula,
        "costs_enabled": bool(config.costs.enabled),
        "cost_bps": float(cost_bps),
        "annualization_periods": int(config.primary_payoff.annualization_periods),
        "horizon_trading_days": int(config.primary_payoff.horizon_trading_days),
        "overlapping_labels": bool(config.primary_payoff.overlapping_labels),
        "research_proxy_not_trade_pnl": bool(config.primary_payoff.report_as_research_proxy),
        "strategy_universe_locked": True,
        "strategy_universe": list(BACKTEST_STRATEGY_UNIVERSE),
        "n_rows": int(len(panel)),
        "n_eligible_rows": int(eligible.sum()),
        "n_ineligible_rows": int((~eligible).sum()),
        "n_strategies": int(panel["strategy_name"].nunique()) if len(panel) else 0,
        "strategy_row_counts": strategy_counts,
        "eligible_row_counts": eligible_counts,
        "exclusion_reason_counts": exclusion_counts,
        "signal_observation_date_min": None if pd.isna(signal_min) else str(pd.Timestamp(signal_min).date()),
        "signal_observation_date_max": None if pd.isna(signal_max) else str(pd.Timestamp(signal_max).date()),
        "target_trade_date_min": None if pd.isna(trade_min) else str(pd.Timestamp(trade_min).date()),
        "target_trade_date_max": None if pd.isna(trade_max) else str(pd.Timestamp(trade_max).date()),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "annualized_metrics_interpretation": (
            "approximate; observations are not independent daily returns"
        ),
        "limitations": [
            "Primary payoff is a research-layer forward VRP payoff proxy.",
            "This is not executed option-trading PnL.",
            "The payoff label is a 22-trading-day forward ex-post realised outcome.",
            "Forward labels may overlap.",
            "Annualised metrics are approximate and mainly useful for strategy comparison.",
        ],
    }
    metadata.update(_count_no_lookahead_violations(panel))

    return _json_ready(metadata)


def write_backtest_outputs(
    panel: pd.DataFrame,
    *,
    output_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    force: bool = False,
) -> None:
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)

    _ensure_output_can_be_written(output_path, metadata_path, force=force)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(output_path, index=False)
    metadata_path.write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_market_backtest(
    *,
    market: str,
    config: BacktestConfig,
    repo_root: Path,
    strategy: str = "all",
    cost_bps: float | None = None,
    force: bool = False,
    write: bool = True,
) -> BacktestRunResult:
    market = market.upper()
    if market not in SUPPORTED_MARKETS:
        raise VectorizedBacktestError(
            f"Unsupported market {market!r}. Expected one of {SUPPORTED_MARKETS}."
        )

    strategy = _normalize_strategy(strategy)

    if write and strategy != "all":
        raise VectorizedBacktestError(
            "Writing a single-strategy panel to the canonical Phase 10 output path "
            "is disabled. Use --dry-run for single-strategy inspection, or run "
            "--strategy all to write data/processed/*_backtest_panel.parquet."
        )

    config = _resolve_config_paths(config, repo_root=repo_root)
    assert_strategy_universe_locked(config.strategy_universe)

    effective_cost_bps = (
        float(config.costs.default_cost_bps)
        if cost_bps is None
        else float(cost_bps)
    )
    if effective_cost_bps < 0:
        raise VectorizedBacktestError(
            f"cost_bps must be non-negative. Got {effective_cost_bps}."
        )

    output_path = get_market_output_path(config, market, "backtest_panel")
    metadata_path = get_market_output_path(config, market, "metadata")

    if write:
        _ensure_output_can_be_written(output_path, metadata_path, force=force)

    full_panel = build_research_backtest_panel(
        market,
        config,
        cost_bps=effective_cost_bps,
    )
    panel = _filter_strategy(full_panel, strategy)

    validate_backtest_panel_integrity(panel, config=config)

    metadata = build_backtest_metadata(
        panel,
        market=market,
        strategy=strategy,
        config=config,
        cost_bps=effective_cost_bps,
        output_path=output_path,
        metadata_path=metadata_path,
    )

    if write:
        write_backtest_outputs(
            panel,
            output_path=output_path,
            metadata_path=metadata_path,
            metadata=metadata,
            force=force,
        )

    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    return BacktestRunResult(
        market=market,
        strategy=strategy,
        output_path=output_path,
        metadata_path=metadata_path,
        n_rows=int(len(panel)),
        n_eligible=int(eligible.sum()),
        n_strategies=int(panel["strategy_name"].nunique()) if len(panel) else 0,
        cost_bps=effective_cost_bps,
        wrote_files=write,
        metadata=metadata,
    )


def run_backtests(
    *,
    markets: Iterable[str],
    config: BacktestConfig,
    repo_root: Path,
    strategy: str = "all",
    cost_bps: float | None = None,
    force: bool = False,
    write: bool = True,
) -> list[BacktestRunResult]:
    results: list[BacktestRunResult] = []

    for market in markets:
        results.append(
            run_market_backtest(
                market=market,
                config=config,
                repo_root=repo_root,
                strategy=strategy,
                cost_bps=cost_bps,
                force=force,
                write=write,
            )
        )

    return results


def render_run_summary(results: Iterable[BacktestRunResult]) -> str:
    lines: list[str] = []

    for result in results:
        status = "wrote" if result.wrote_files else "dry-run"
        lines.append(
            f"[{result.market}] {status} rows={result.n_rows} "
            f"eligible={result.n_eligible} strategies={result.n_strategies} "
            f"cost_bps={result.cost_bps:g}"
        )
        lines.append(f"  panel:    {result.output_path}")
        lines.append(f"  metadata: {result.metadata_path}")

    if not lines:
        return "No backtests were run."

    return "\n".join(lines)