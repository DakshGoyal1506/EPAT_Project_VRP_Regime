from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp.backtest.backtest_config import (
    BacktestConfig,
    get_market_output_path,
)
from vrp.backtest.metrics import (
    build_availability_summary,
    build_strategy_metric_table,
    compute_cte,
    compute_equity_curve,
    compute_strategy_metrics,
    get_metrics_metadata,
)
from vrp.backtest.vectorized_engine import resolve_markets


SUPPORTED_MARKETS: tuple[str, ...] = ("US", "INDIA")

REPORT_LIMITATIONS: tuple[str, ...] = (
    "Primary payoff is a research-layer forward VRP payoff proxy.",
    "This is not executed option-trading PnL.",
    "The payoff label is a 22-trading-day forward ex-post realised outcome.",
    "Forward labels may overlap.",
    "Annualised metrics are approximate and mainly useful for strategy comparison.",
)

VISUAL_INTERPRETATION_WARNING = (
    "Cumulative curves are additive research proxy sums over overlapping forward labels. "
    "They are not executable account equity curves."
)


class BacktestDiagnosticsError(ValueError):
    """Raised when Phase 10 report diagnostics cannot be generated."""


@dataclass(frozen=True)
class BacktestDiagnosticsResult:
    table_paths: dict[str, Path]
    figure_paths: dict[str, Path]
    metadata_path: Path


def _resolve_path(repo_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]

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


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _read_panel(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise BacktestDiagnosticsError(f"Backtest panel does not exist: {path}")

    if path.suffix.lower() != ".parquet":
        raise BacktestDiagnosticsError(
            f"Backtest panel must be parquet. Got: {path}"
        )

    df = pd.read_parquet(path)

    required = {
        "market",
        "strategy_name",
        "signal_observation_date",
        "target_trade_date",
        "outcome_label_date",
        "is_backtest_eligible",
        "exclusion_reason",
        "gross_return_proxy",
        "net_return_proxy",
        "target_exposure_for_backtest",
        "delta_exposure",
        "cost_proxy",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise BacktestDiagnosticsError(
            f"Backtest panel {path} missing required columns: {missing}"
        )

    return _normalize_panel_dates(df)


def _normalize_panel_dates(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    for col in ("signal_observation_date", "target_trade_date", "outcome_label_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()

    return out


def load_backtest_panels(
    config: BacktestConfig,
    *,
    repo_root: Path,
    markets: Iterable[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for market in markets:
        market_key = market.upper()
        path = _resolve_path(
            repo_root,
            get_market_output_path(config, market_key, "backtest_panel"),
        )
        frame = _read_panel(path)
        frames.append(frame)

    if not frames:
        raise BacktestDiagnosticsError("No backtest panels loaded.")

    return pd.concat(frames, ignore_index=True)


def build_backtest_summary_table(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    summary = build_strategy_metric_table(
        panel,
        annualization_periods=config.primary_payoff.annualization_periods,
        horizon_trading_days=config.primary_payoff.horizon_trading_days,
    )

    if summary.empty:
        return summary

    front_cols = [
        "market",
        "strategy_name",
        "n_obs",
        "n_eligible",
        "availability_rate",
        "total_gross_return",
        "total_cost",
        "total_return_proxy",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "mean_exposure",
        "mean_abs_exposure",
        "sum_abs_exposure",
        "return_per_abs_exposure",
        "drawdown_per_abs_exposure",
        "return_to_drawdown",
        "turnover",
        "uses_overlapping_forward_labels",
        "horizon_trading_days",
        "research_proxy_not_trade_pnl",
        "annualized_metrics_interpretation",
    ]

    ordered = [col for col in front_cols if col in summary.columns]
    remaining = [col for col in summary.columns if col not in ordered]

    return summary.loc[:, ordered + remaining]


def build_backtest_by_strategy_year_table(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    out = panel.copy()
    out["target_trade_year"] = pd.to_datetime(
        out["target_trade_date"],
        errors="coerce",
    ).dt.year

    out = out.dropna(subset=["target_trade_year"]).copy()
    out["target_trade_year"] = out["target_trade_year"].astype(int)

    rows: list[dict[str, Any]] = []

    for (market, strategy_name, year), group in out.groupby(
        ["market", "strategy_name", "target_trade_year"],
        dropna=False,
        sort=True,
    ):
        row = {
            "market": market,
            "strategy_name": strategy_name,
            "year": int(year),
        }
        row.update(
            compute_strategy_metrics(
                group,
                annualization_periods=config.primary_payoff.annualization_periods,
                horizon_trading_days=config.primary_payoff.horizon_trading_days,
            )
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["market", "strategy_name", "year"]
    ).reset_index(drop=True)


def get_common_start_dates(
    panel: pd.DataFrame,
    *,
    eligible_col: str = "is_backtest_eligible",
) -> dict[str, str]:
    """
    For each market, find the latest first eligible target_trade_date
    across strategies. This creates a fair visual comparison window.
    """
    if panel.empty:
        return {}

    required = {"market", "strategy_name", "target_trade_date", eligible_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise BacktestDiagnosticsError(
            f"Cannot compute common-start dates. Missing columns: {missing}"
        )

    out = panel.copy()
    out["market"] = out["market"].astype(str).str.upper()
    out["target_trade_date"] = pd.to_datetime(
        out["target_trade_date"],
        errors="coerce",
    ).dt.normalize()

    eligible = out[eligible_col].fillna(False).astype(bool)
    out = out.loc[eligible & out["target_trade_date"].notna()].copy()

    if out.empty:
        return {}

    first_dates = (
        out.groupby(["market", "strategy_name"], dropna=False)["target_trade_date"]
        .min()
        .reset_index()
    )

    common_dates = (
        first_dates.groupby("market", dropna=False)["target_trade_date"]
        .max()
        .sort_index()
    )

    return {
        str(market): str(pd.Timestamp(date).date())
        for market, date in common_dates.items()
        if pd.notna(date)
    }


def build_common_start_panel(
    panel: pd.DataFrame,
    *,
    eligible_col: str = "is_backtest_eligible",
) -> pd.DataFrame:
    """
    Restrict each market to the date from which all available strategy
    histories can be compared fairly.
    """
    common_dates = get_common_start_dates(panel, eligible_col=eligible_col)

    out = panel.copy()
    out["market"] = out["market"].astype(str).str.upper()
    out["target_trade_date"] = pd.to_datetime(
        out["target_trade_date"],
        errors="coerce",
    ).dt.normalize()

    frames: list[pd.DataFrame] = []

    for market, start_date in common_dates.items():
        start_ts = pd.Timestamp(start_date)
        market_frame = out.loc[
            out["market"].eq(market)
            & out["target_trade_date"].notna()
            & (out["target_trade_date"] >= start_ts)
        ].copy()
        market_frame["common_start_date"] = str(start_ts.date())
        frames.append(market_frame)

    if not frames:
        empty = out.iloc[0:0].copy()
        empty["common_start_date"] = pd.Series(dtype="object")
        return empty

    return pd.concat(frames, ignore_index=True)


def build_common_start_summary_table(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    """
    Recompute metrics after applying each market's common-start date.
    This avoids visual/performance bias from different strategy start dates.
    """
    common_panel = build_common_start_panel(panel)

    if common_panel.empty:
        return pd.DataFrame()

    summary = build_strategy_metric_table(
        common_panel,
        annualization_periods=config.primary_payoff.annualization_periods,
        horizon_trading_days=config.primary_payoff.horizon_trading_days,
    )

    common_dates = get_common_start_dates(panel)
    summary["common_start_date"] = summary["market"].astype(str).str.upper().map(common_dates)

    front_cols = [
        "market",
        "strategy_name",
        "common_start_date",
        "n_obs",
        "n_eligible",
        "availability_rate",
        "total_gross_return",
        "total_cost",
        "total_return_proxy",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "hit_rate",
        "mean_abs_exposure",
        "sum_abs_exposure",
        "return_per_abs_exposure",
        "drawdown_per_abs_exposure",
        "return_to_drawdown",
        "turnover",
        "uses_overlapping_forward_labels",
        "horizon_trading_days",
        "research_proxy_not_trade_pnl",
        "annualized_metrics_interpretation",
    ]

    ordered = [col for col in front_cols if col in summary.columns]
    remaining = [col for col in summary.columns if col not in ordered]

    return summary.loc[:, ordered + remaining].sort_values(
        ["market", "strategy_name"]
    ).reset_index(drop=True)


def build_tail_summary_table(
    panel: pd.DataFrame,
    *,
    return_col: str = "net_return_proxy",
    eligible_col: str = "is_backtest_eligible",
) -> pd.DataFrame:
    """
    Summarise strategy-level tails because pooled histograms hide
    the economically important loss distribution.
    """
    required = {"market", "strategy_name", return_col, eligible_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise BacktestDiagnosticsError(
            f"Cannot build tail summary. Missing columns: {missing}"
        )

    rows: list[dict[str, Any]] = []

    for (market, strategy_name), group in panel.groupby(
        ["market", "strategy_name"],
        dropna=False,
        sort=True,
    ):
        eligible = group[eligible_col].fillna(False).astype(bool)
        returns = pd.to_numeric(
            group.loc[eligible, return_col],
            errors="coerce",
        ).dropna()

        if returns.empty:
            rows.append(
                {
                    "market": market,
                    "strategy_name": strategy_name,
                    "n_eligible": 0,
                    "p01": float("nan"),
                    "p05": float("nan"),
                    "p50": float("nan"),
                    "p95": float("nan"),
                    "p99": float("nan"),
                    "worst_return": float("nan"),
                    "best_return": float("nan"),
                    "cte_95": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "market": market,
                "strategy_name": strategy_name,
                "n_eligible": int(len(returns)),
                "p01": float(returns.quantile(0.01)),
                "p05": float(returns.quantile(0.05)),
                "p50": float(returns.quantile(0.50)),
                "p95": float(returns.quantile(0.95)),
                "p99": float(returns.quantile(0.99)),
                "worst_return": float(returns.min()),
                "best_return": float(returns.max()),
                "cte_95": compute_cte(returns, alpha=0.95),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["market", "strategy_name"]
    ).reset_index(drop=True)


def build_crisis_window_performance_table(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for market, subperiods in config.robustness.subperiods.items():
        market_panel = panel.loc[panel["market"].astype(str).str.upper().eq(market)].copy()

        if market_panel.empty:
            continue

        trade_dates = pd.to_datetime(
            market_panel["target_trade_date"],
            errors="coerce",
        )

        for start, end, label in subperiods:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)

            window_panel = market_panel.loc[
                (trade_dates >= start_ts) & (trade_dates <= end_ts)
            ].copy()

            for strategy_name in config.strategy_universe:
                group = window_panel.loc[
                    window_panel["strategy_name"].astype(str).eq(strategy_name)
                ].copy()

                row = {
                    "market": market,
                    "subperiod": label,
                    "start_date": str(start_ts.date()),
                    "end_date": str(end_ts.date()),
                    "strategy_name": strategy_name,
                }
                row.update(
                    compute_strategy_metrics(
                        group,
                        annualization_periods=config.primary_payoff.annualization_periods,
                        horizon_trading_days=config.primary_payoff.horizon_trading_days,
                    )
                )
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["market", "subperiod", "strategy_name"]
    ).reset_index(drop=True)


def build_no_lookahead_audit_table(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (market, strategy_name), group in panel.groupby(
        ["market", "strategy_name"],
        dropna=False,
        sort=True,
    ):
        eligible = group["is_backtest_eligible"].fillna(False).astype(bool)

        signal_dates = pd.to_datetime(
            group.loc[eligible, "signal_observation_date"],
            errors="coerce",
        )
        trade_dates = pd.to_datetime(
            group.loc[eligible, "target_trade_date"],
            errors="coerce",
        )
        outcome_dates = pd.to_datetime(
            group.loc[eligible, "outcome_label_date"],
            errors="coerce",
        )

        if eligible.any():
            n_target_not_after_signal = int((trade_dates <= signal_dates).sum())
            n_outcome_not_equal_signal = int((outcome_dates != signal_dates).sum())
            n_missing_net_return = int(
                pd.to_numeric(
                    group.loc[eligible, "net_return_proxy"],
                    errors="coerce",
                ).isna().sum()
            )
            n_missing_backtest_exposure = int(
                pd.to_numeric(
                    group.loc[eligible, "target_exposure_for_backtest"],
                    errors="coerce",
                ).isna().sum()
            )
        else:
            n_target_not_after_signal = 0
            n_outcome_not_equal_signal = 0
            n_missing_net_return = 0
            n_missing_backtest_exposure = 0

        rows.append(
            {
                "market": market,
                "strategy_name": strategy_name,
                "n_rows": int(len(group)),
                "n_eligible": int(eligible.sum()),
                "outcome_alignment": config.primary_payoff.outcome_alignment,
                "n_target_not_after_signal_violations": n_target_not_after_signal,
                "n_outcome_not_equal_signal_date_violations": (
                    n_outcome_not_equal_signal
                    if config.primary_payoff.outcome_alignment == "signal_observation_date"
                    else 0
                ),
                "n_missing_net_return_on_eligible_rows": n_missing_net_return,
                "n_missing_backtest_exposure_on_eligible_rows": n_missing_backtest_exposure,
                "passes_no_lookahead_audit": (
                    n_target_not_after_signal == 0
                    and (
                        config.primary_payoff.outcome_alignment != "signal_observation_date"
                        or n_outcome_not_equal_signal == 0
                    )
                    and n_missing_net_return == 0
                    and n_missing_backtest_exposure == 0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["market", "strategy_name"]
    ).reset_index(drop=True)


def _plot_market_equity_curves(
    panel: pd.DataFrame,
    *,
    market: str,
    output_path: Path,
) -> Path:
    market_panel = panel.loc[panel["market"].astype(str).str.upper().eq(market)].copy()

    if market_panel.empty:
        raise BacktestDiagnosticsError(f"No rows available for market {market}.")

    plt.figure(figsize=(12, 7))

    plotted = False
    for strategy_name, group in market_panel.groupby("strategy_name", sort=True):
        curve = compute_equity_curve(group)

        if curve.empty:
            continue

        plt.plot(
            curve["target_trade_date"],
            curve["equity_curve"],
            label=strategy_name,
        )
        plotted = True

    if not plotted:
        raise BacktestDiagnosticsError(
            f"No eligible equity curve observations for market {market}."
        )

    plt.title(f"{market} Phase 10 Research Proxy Equity Curves")
    plt.xlabel("Target trade date")
    plt.ylabel("Cumulative net return proxy")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def _plot_market_common_start_equity_curves(
    panel: pd.DataFrame,
    *,
    market: str,
    output_path: Path,
) -> Path:
    common_panel = build_common_start_panel(panel)
    market_panel = common_panel.loc[
        common_panel["market"].astype(str).str.upper().eq(market)
    ].copy()

    if market_panel.empty:
        raise BacktestDiagnosticsError(
            f"No common-start rows available for market {market}."
        )

    common_start_values = sorted(
        market_panel["common_start_date"].dropna().astype(str).unique().tolist()
    )
    common_start_label = common_start_values[0] if common_start_values else "unknown"

    plt.figure(figsize=(12, 7))

    plotted = False
    for strategy_name, group in market_panel.groupby("strategy_name", sort=True):
        curve = compute_equity_curve(group)

        if curve.empty:
            continue

        plt.plot(
            curve["target_trade_date"],
            curve["equity_curve"],
            label=strategy_name,
        )
        plotted = True

    if not plotted:
        raise BacktestDiagnosticsError(
            f"No eligible common-start equity observations for market {market}."
        )

    plt.title(
        f"{market} Phase 10 Common-Start Research Proxy Equity Curves "
        f"(from {common_start_label})"
    )
    plt.xlabel("Target trade date")
    plt.ylabel("Cumulative net return proxy")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def _plot_market_drawdowns(
    panel: pd.DataFrame,
    *,
    market: str,
    output_path: Path,
) -> Path:
    market_panel = panel.loc[panel["market"].astype(str).str.upper().eq(market)].copy()

    if market_panel.empty:
        raise BacktestDiagnosticsError(f"No rows available for market {market}.")

    plt.figure(figsize=(12, 7))

    plotted = False
    for strategy_name, group in market_panel.groupby("strategy_name", sort=True):
        curve = compute_equity_curve(group)

        if curve.empty:
            continue

        plt.plot(
            curve["target_trade_date"],
            curve["drawdown"],
            label=strategy_name,
        )
        plotted = True

    if not plotted:
        raise BacktestDiagnosticsError(
            f"No eligible drawdown observations for market {market}."
        )

    plt.title(f"{market} Phase 10 Research Proxy Drawdowns")
    plt.xlabel("Target trade date")
    plt.ylabel("Drawdown from cumulative proxy peak")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def _plot_market_return_distribution(
    panel: pd.DataFrame,
    *,
    market: str,
    output_path: Path,
) -> Path:
    market_panel = panel.loc[panel["market"].astype(str).str.upper().eq(market)].copy()

    if market_panel.empty:
        raise BacktestDiagnosticsError(f"No rows available for market {market}.")

    eligible = market_panel["is_backtest_eligible"].fillna(False).astype(bool)
    returns = pd.to_numeric(
        market_panel.loc[eligible, "net_return_proxy"],
        errors="coerce",
    ).dropna()

    if returns.empty:
        raise BacktestDiagnosticsError(
            f"No eligible return observations for market {market}."
        )

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50)
    plt.title(f"{market} Phase 10 Net Return Proxy Distribution")
    plt.xlabel("Net return proxy")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def write_backtest_diagnostic_tables(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
    table_dir: Path,
) -> dict[str, Path]:
    _ensure_dir(table_dir)

    summary = build_backtest_summary_table(panel, config=config)
    common_start_summary = build_common_start_summary_table(panel, config=config)
    tail_summary = build_tail_summary_table(panel)
    by_year = build_backtest_by_strategy_year_table(panel, config=config)
    crisis = build_crisis_window_performance_table(panel, config=config)
    availability = build_availability_summary(panel)
    no_lookahead = build_no_lookahead_audit_table(panel, config=config)

    table_paths = {
        "backtest_summary": _write_csv(
            summary,
            table_dir / "backtest_summary.csv",
        ),
        "backtest_common_start_summary": _write_csv(
            common_start_summary,
            table_dir / "backtest_common_start_summary.csv",
        ),
        "backtest_tail_summary": _write_csv(
            tail_summary,
            table_dir / "backtest_tail_summary.csv",
        ),
        "backtest_by_strategy_year": _write_csv(
            by_year,
            table_dir / "backtest_by_strategy_year.csv",
        ),
        "crisis_window_performance": _write_csv(
            crisis,
            table_dir / "crisis_window_performance.csv",
        ),
        "backtest_availability_summary": _write_csv(
            availability,
            table_dir / "backtest_availability_summary.csv",
        ),
        "backtest_no_lookahead_audit": _write_csv(
            no_lookahead,
            table_dir / "backtest_no_lookahead_audit.csv",
        ),
    }

    return table_paths


def write_backtest_diagnostic_figures(
    panel: pd.DataFrame,
    *,
    markets: Sequence[str],
    figure_dir: Path,
) -> dict[str, Path]:
    _ensure_dir(figure_dir)

    figure_paths: dict[str, Path] = {}

    for market in markets:
        market_lower = market.lower()

        figure_paths[f"equity_curves_{market_lower}"] = _plot_market_equity_curves(
            panel,
            market=market,
            output_path=figure_dir / f"equity_curves_{market_lower}.png",
        )
        figure_paths[f"equity_curves_common_start_{market_lower}"] = (
            _plot_market_common_start_equity_curves(
                panel,
                market=market,
                output_path=figure_dir / f"equity_curves_common_start_{market_lower}.png",
            )
        )
        figure_paths[f"drawdowns_{market_lower}"] = _plot_market_drawdowns(
            panel,
            market=market,
            output_path=figure_dir / f"drawdowns_{market_lower}.png",
        )
        figure_paths[f"return_distribution_{market_lower}"] = (
            _plot_market_return_distribution(
                panel,
                market=market,
                output_path=figure_dir / f"return_distribution_{market_lower}.png",
            )
        )

    return figure_paths


def build_report_metadata(
    panel: pd.DataFrame,
    *,
    config: BacktestConfig,
    markets: Sequence[str],
    table_paths: dict[str, Path],
    figure_paths: dict[str, Path],
) -> dict[str, Any]:
    eligible = panel["is_backtest_eligible"].fillna(False).astype(bool)

    no_lookahead = build_no_lookahead_audit_table(panel, config=config)
    no_lookahead_pass = bool(no_lookahead["passes_no_lookahead_audit"].all())
    common_start_dates = get_common_start_dates(panel)

    return {
        "phase": "phase_10",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "markets": list(markets),
        "n_rows": int(len(panel)),
        "n_eligible_rows": int(eligible.sum()),
        "n_ineligible_rows": int((~eligible).sum()),
        "strategy_universe": list(config.strategy_universe),
        "payoff_type": config.primary_payoff.name,
        "payoff_label": config.primary_payoff.label_col,
        "label_role": config.primary_payoff.label_role,
        "outcome_alignment": config.primary_payoff.outcome_alignment,
        "annualization_periods": int(config.primary_payoff.annualization_periods),
        "horizon_trading_days": int(config.primary_payoff.horizon_trading_days),
        "overlapping_labels": bool(config.primary_payoff.overlapping_labels),
        "research_proxy_not_trade_pnl": bool(config.primary_payoff.report_as_research_proxy),
        "visual_interpretation_warning": VISUAL_INTERPRETATION_WARNING,
        "common_start_dates": common_start_dates,
        "no_lookahead_audit_passed": no_lookahead_pass,
        "metrics_metadata": get_metrics_metadata(
            horizon_trading_days=config.primary_payoff.horizon_trading_days,
        ),
        "limitations": list(REPORT_LIMITATIONS),
        "table_paths": {key: str(path) for key, path in table_paths.items()},
        "figure_paths": {key: str(path) for key, path in figure_paths.items()},
    }


def generate_backtest_diagnostics(
    *,
    config: BacktestConfig,
    repo_root: Path,
    market: str = "ALL",
) -> BacktestDiagnosticsResult:
    markets = resolve_markets(market)

    panel = load_backtest_panels(
        config,
        repo_root=repo_root,
        markets=markets,
    )

    table_dir = _resolve_path(repo_root, config.reporting["table_dir"])
    figure_dir = _resolve_path(repo_root, config.reporting["figure_dir"])

    table_paths = write_backtest_diagnostic_tables(
        panel,
        config=config,
        table_dir=table_dir,
    )
    figure_paths = write_backtest_diagnostic_figures(
        panel,
        markets=markets,
        figure_dir=figure_dir,
    )

    metadata = build_report_metadata(
        panel,
        config=config,
        markets=markets,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )
    metadata_path = _write_json(
        metadata,
        table_dir / "backtest_metadata.json",
    )

    return BacktestDiagnosticsResult(
        table_paths=table_paths,
        figure_paths=figure_paths,
        metadata_path=metadata_path,
    )


def render_diagnostics_summary(result: BacktestDiagnosticsResult) -> str:
    lines = ["Phase 10 diagnostic reports written."]

    lines.append("Tables:")
    for name, path in result.table_paths.items():
        lines.append(f"  {name}: {path}")

    lines.append("Figures:")
    for name, path in result.figure_paths.items():
        lines.append(f"  {name}: {path}")

    lines.append(f"Metadata: {result.metadata_path}")

    return "\n".join(lines)