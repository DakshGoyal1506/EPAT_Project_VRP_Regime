from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from vrp.backtest.backtest_config import BacktestConfig
from vrp.backtest.metrics import build_strategy_metric_table, compute_strategy_metrics
from vrp.backtest.payoff_proxies import build_research_backtest_panel
from vrp.backtest.tradable_proxy_detector import (
    detect_tradable_proxy_data,
    write_tradable_proxy_detection_report,
)
from vrp.backtest.vectorized_engine import resolve_markets


class RobustnessError(ValueError):
    """Raised when Phase 10 robustness cannot be generated."""


@dataclass(frozen=True)
class RobustnessRunResult:
    test_name: str
    status: str
    output_paths: dict[str, Path]
    message: str


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
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


def _table_dir(config: BacktestConfig, repo_root: Path) -> Path:
    return _resolve_path(repo_root, config.reporting["table_dir"])


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return path


def _load_backtest_panel_or_build(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str,
    cost_bps: float | None = None,
) -> pd.DataFrame:
    """
    Prefer existing Phase 10 backtest panel. If missing, build in memory.

    This function does not write panels.
    """
    market_key = market.upper()
    config_resolved = _resolve_config_paths(config, repo_root=repo_root)

    panel_path = Path(config_resolved.output_files[market_key]["backtest_panel"])

    if panel_path.exists():
        return pd.read_parquet(panel_path)

    return build_research_backtest_panel(
        market_key,
        config_resolved,
        cost_bps=cost_bps,
    )


def _add_metric_context(
    table: pd.DataFrame,
    *,
    robustness_test: str,
    market: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = table.copy()
    out.insert(0, "robustness_test", robustness_test)
    out.insert(1, "market_tested", market)

    if extra:
        insert_at = 2
        for key, value in extra.items():
            out.insert(insert_at, key, value)
            insert_at += 1

    return out


def run_cost_sensitivity_robustness(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
    cost_bps_grid: Iterable[float] | None = None,
    force: bool = False,
) -> RobustnessRunResult:
    """
    Rebuild panels at each cost level and recompute strategy metrics.

    This changes only transaction-cost assumptions. Strategy signals and payoff
    labels remain unchanged.
    """
    repo_root = Path(repo_root)
    markets = resolve_markets(market)
    config_resolved = _resolve_config_paths(config, repo_root=repo_root)

    grid = (
        tuple(float(value) for value in cost_bps_grid)
        if cost_bps_grid is not None
        else tuple(float(value) for value in config.robustness.cost_bps_grid)
    )

    if any(value < 0 for value in grid):
        raise RobustnessError(f"Cost grid must be non-negative. Got {grid}.")

    rows: list[pd.DataFrame] = []

    for market_code in markets:
        for cost_bps in grid:
            panel = build_research_backtest_panel(
                market_code,
                config_resolved,
                cost_bps=cost_bps,
            )
            metrics = build_strategy_metric_table(
                panel,
                annualization_periods=config.primary_payoff.annualization_periods,
                horizon_trading_days=config.primary_payoff.horizon_trading_days,
            )
            metrics = _add_metric_context(
                metrics,
                robustness_test="cost_sensitivity",
                market=market_code,
                extra={"cost_bps": cost_bps},
            )
            rows.append(metrics)

    if rows:
        out = pd.concat(rows, ignore_index=True)
    else:
        out = pd.DataFrame()

    output_path = _table_dir(config, repo_root) / "robustness_cost_sensitivity.csv"

    if output_path.exists() and not force:
        raise RobustnessError(f"Output exists. Use --force to overwrite: {output_path}")

    _write_csv(out, output_path)

    return RobustnessRunResult(
        test_name="costs",
        status="completed",
        output_paths={"cost_sensitivity": output_path},
        message=f"Wrote cost sensitivity table with {len(out)} rows.",
    )


def run_subperiod_robustness(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
    force: bool = False,
) -> RobustnessRunResult:
    """
    Compute metrics for pre-specified subperiod windows.

    Uses existing backtest panel when present; otherwise builds the base-cost
    panel in memory.
    """
    repo_root = Path(repo_root)
    markets = resolve_markets(market)

    rows: list[dict[str, Any]] = []

    for market_code in markets:
        panel = _load_backtest_panel_or_build(
            config=config,
            repo_root=repo_root,
            market=market_code,
            cost_bps=config.costs.default_cost_bps,
        )

        if panel.empty:
            continue

        panel = panel.copy()
        panel["target_trade_date"] = pd.to_datetime(
            panel["target_trade_date"],
            errors="coerce",
        ).dt.normalize()

        subperiods = config.robustness.subperiods.get(market_code, ())

        for start_date, end_date, subperiod_name in subperiods:
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)

            window_panel = panel.loc[
                panel["target_trade_date"].notna()
                & (panel["target_trade_date"] >= start_ts)
                & (panel["target_trade_date"] <= end_ts)
            ].copy()

            for strategy_name in config.strategy_universe:
                strategy_panel = window_panel.loc[
                    window_panel["strategy_name"].astype(str).eq(strategy_name)
                ].copy()

                metric_row = compute_strategy_metrics(
                    strategy_panel,
                    annualization_periods=config.primary_payoff.annualization_periods,
                    horizon_trading_days=config.primary_payoff.horizon_trading_days,
                )

                metric_row.update(
                    {
                        "robustness_test": "subperiod",
                        "market": market_code,
                        "strategy_name": strategy_name,
                        "subperiod": subperiod_name,
                        "subperiod_start": str(start_ts.date()),
                        "subperiod_end": str(end_ts.date()),
                    }
                )
                rows.append(metric_row)

    out = pd.DataFrame(rows)

    if not out.empty:
        front_cols = [
            "robustness_test",
            "market",
            "strategy_name",
            "subperiod",
            "subperiod_start",
            "subperiod_end",
            "n_obs",
            "n_eligible",
            "availability_rate",
            "total_return_proxy",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
            "hit_rate",
            "mean_abs_exposure",
            "return_per_abs_exposure",
            "return_to_drawdown",
            "uses_overlapping_forward_labels",
            "horizon_trading_days",
            "research_proxy_not_trade_pnl",
            "annualized_metrics_interpretation",
        ]
        ordered = [col for col in front_cols if col in out.columns]
        remaining = [col for col in out.columns if col not in ordered]
        out = out.loc[:, ordered + remaining].sort_values(
            ["market", "strategy_name", "subperiod"]
        ).reset_index(drop=True)

    output_path = _table_dir(config, repo_root) / "robustness_subperiods.csv"

    if output_path.exists() and not force:
        raise RobustnessError(f"Output exists. Use --force to overwrite: {output_path}")

    _write_csv(out, output_path)

    return RobustnessRunResult(
        test_name="subperiod",
        status="completed",
        output_paths={"subperiod": output_path},
        message=f"Wrote subperiod robustness table with {len(out)} rows.",
    )


def write_weekly_rebalance_skip_report(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
    force: bool = False,
) -> RobustnessRunResult:
    """
    Weekly rebalance is intentionally skip-safe by default.

    The approved rule requires carrying exposure only across eligible rows and
    never across unavailable rows or missing outcome labels. That is safer to
    implement as a separate audited module later.
    """
    output_path = _table_dir(config, repo_root) / "robustness_weekly_rebalance_skipped.json"

    if output_path.exists() and not force:
        raise RobustnessError(f"Output exists. Use --force to overwrite: {output_path}")

    payload = {
        "phase": "phase_10",
        "robustness_test": "weekly_rebalance",
        "status": "skipped",
        "market": market,
        "reason": (
            "Weekly rebalance robustness is optional and was skipped by default. "
            "A safe implementation must allow exposure changes only on the first "
            "eligible target_trade_date of each calendar week, carry exposure only "
            "across eligible rows, and never carry across unavailable strategy rows "
            "or missing outcome labels."
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    _write_json(payload, output_path)

    return RobustnessRunResult(
        test_name="weekly",
        status="skipped",
        output_paths={"weekly_rebalance_skip": output_path},
        message="Wrote weekly rebalance skipped report.",
    )


def run_tradable_proxy_detection(
    *,
    config: BacktestConfig,
    repo_root: Path,
    force: bool = False,
) -> RobustnessRunResult:
    """
    Detect existing tradable proxy data only. No downloads.
    """
    output_path = _table_dir(config, repo_root) / "tradable_proxy_detection.json"

    if output_path.exists() and not force:
        raise RobustnessError(f"Output exists. Use --force to overwrite: {output_path}")

    result = detect_tradable_proxy_data(repo_root)
    write_tradable_proxy_detection_report(result, output_path)

    return RobustnessRunResult(
        test_name="tradable_proxy",
        status=result.status,
        output_paths={"tradable_proxy_detection": output_path},
        message=result.reason,
    )


def write_robustness_metadata(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str,
    results: list[RobustnessRunResult],
    force: bool = True,
) -> Path:
    output_path = _table_dir(config, repo_root) / "robustness_metadata.json"

    if output_path.exists() and not force:
        raise RobustnessError(f"Output exists. Use --force to overwrite: {output_path}")

    payload = {
        "phase": "phase_10",
        "market": market,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "tests": [
            {
                "test_name": result.test_name,
                "status": result.status,
                "message": result.message,
                "output_paths": {
                    key: str(path)
                    for key, path in result.output_paths.items()
                },
            }
            for result in results
        ],
        "rules": {
            "no_new_data_downloads": True,
            "no_new_strategies": True,
            "no_msvol_strategy_use": True,
            "tradable_proxy_detection_only": True,
            "weekly_rebalance_skip_safe_by_default": True,
            "research_proxy_not_trade_pnl": True,
            "overlapping_labels": True,
        },
    }

    return _write_json(payload, output_path)


def run_robustness_suite(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
    test: str = "all",
    force: bool = False,
) -> list[RobustnessRunResult]:
    test = test.lower()

    allowed = {"costs", "subperiod", "weekly", "tradable_proxy", "all"}
    if test not in allowed:
        raise RobustnessError(f"Unknown robustness test {test!r}. Allowed: {sorted(allowed)}")

    results: list[RobustnessRunResult] = []

    if test in {"costs", "all"}:
        results.append(
            run_cost_sensitivity_robustness(
                config=config,
                repo_root=repo_root,
                market=market,
                force=force,
            )
        )

    if test in {"subperiod", "all"}:
        results.append(
            run_subperiod_robustness(
                config=config,
                repo_root=repo_root,
                market=market,
                force=force,
            )
        )

    if test in {"weekly", "all"}:
        results.append(
            write_weekly_rebalance_skip_report(
                config=config,
                repo_root=repo_root,
                market=market,
                force=force,
            )
        )

    if test in {"tradable_proxy", "all"}:
        results.append(
            run_tradable_proxy_detection(
                config=config,
                repo_root=repo_root,
                force=force,
            )
        )

    write_robustness_metadata(
        config=config,
        repo_root=repo_root,
        market=market,
        results=results,
        force=True,
    )

    return results


def render_robustness_summary(results: Iterable[RobustnessRunResult]) -> str:
    lines: list[str] = []

    for result in results:
        lines.append(f"[{result.test_name}] {result.status}: {result.message}")
        for name, path in result.output_paths.items():
            lines.append(f"  {name}: {path}")

    if not lines:
        return "No robustness tests were run."

    return "\n".join(lines)