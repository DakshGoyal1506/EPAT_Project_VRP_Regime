from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ANNUALIZATION_PERIODS = 252
DEFAULT_HORIZON_TRADING_DAYS = 22

ANNUALIZED_METRICS_INTERPRETATION = (
    "approximate; observations are not independent daily returns"
)


class BacktestMetricsError(ValueError):
    """Raised when Phase 10 performance metric computation is invalid."""


@dataclass(frozen=True)
class MetricsMetadata:
    uses_overlapping_forward_labels: bool = True
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS
    annualized_metrics_interpretation: str = ANNUALIZED_METRICS_INTERPRETATION
    research_proxy_not_trade_pnl: bool = True


def get_metrics_metadata(
    *,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
) -> dict[str, Any]:
    return asdict(
        MetricsMetadata(
            uses_overlapping_forward_labels=True,
            horizon_trading_days=int(horizon_trading_days),
            annualized_metrics_interpretation=ANNUALIZED_METRICS_INTERPRETATION,
            research_proxy_not_trade_pnl=True,
        )
    )


def _require_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    label: str,
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise BacktestMetricsError(f"{label} missing required columns: {missing}")


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    eps: float = 1e-12,
) -> float:
    if not np.isfinite(numerator):
        return float("nan")
    if not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) <= eps:
        return float("nan")
    return float(numerator / denominator)


def _as_bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _prepare_eligible_frame(
    df: pd.DataFrame,
    *,
    return_col: str,
    eligible_col: str,
    sort_cols: Sequence[str],
) -> pd.DataFrame:
    _require_columns(
        df,
        [return_col, eligible_col, *sort_cols],
        "backtest panel",
    )

    eligible_mask = _as_bool_mask(df[eligible_col])
    out = df.loc[eligible_mask].copy()

    if out.empty:
        return out

    for col in sort_cols:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out[return_col] = pd.to_numeric(out[return_col], errors="coerce")

    if out[return_col].isna().any():
        n_missing = int(out[return_col].isna().sum())
        raise BacktestMetricsError(
            f"Eligible rows contain missing/non-numeric {return_col}. "
            f"Count={n_missing}."
        )

    out = out.sort_values(list(sort_cols)).reset_index(drop=True)

    return out


def compute_equity_curve(
    df: pd.DataFrame,
    *,
    return_col: str = "net_return_proxy",
    eligible_col: str = "is_backtest_eligible",
    sort_cols: Sequence[str] = ("target_trade_date", "signal_observation_date"),
    equity_col: str = "equity_curve",
    drawdown_col: str = "drawdown",
) -> pd.DataFrame:
    """
    Compute additive research-proxy equity curve.

    This is not compounded trade equity. The input is a VRP payoff proxy, so
    cumulative performance is represented as cumulative sum of proxy returns.
    """
    out = _prepare_eligible_frame(
        df,
        return_col=return_col,
        eligible_col=eligible_col,
        sort_cols=sort_cols,
    )

    if out.empty:
        out[equity_col] = pd.Series(dtype=float)
        out["running_peak"] = pd.Series(dtype=float)
        out[drawdown_col] = pd.Series(dtype=float)
        return out

    returns = out[return_col].astype(float)

    out[equity_col] = returns.cumsum()

    # Include initial zero as the starting peak. This correctly records
    # drawdown if the first eligible observation is negative.
    out["running_peak"] = out[equity_col].cummax().clip(lower=0.0)
    out[drawdown_col] = out[equity_col] - out["running_peak"]

    return out


def compute_max_drawdown(
    returns: pd.Series,
) -> float:
    if returns.empty:
        return float("nan")

    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return float("nan")

    equity = clean.cumsum()
    running_peak = equity.cummax().clip(lower=0.0)
    drawdown = equity - running_peak

    return float(drawdown.min())


def compute_cte(
    returns: pd.Series,
    *,
    alpha: float = 0.95,
) -> float:
    """
    Conditional tail expectation of the loss tail.

    alpha=0.95 means average return in the worst 5% of observations.
    """
    if not 0 < alpha < 1:
        raise BacktestMetricsError(f"alpha must be in (0, 1). Got {alpha}.")

    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return float("nan")

    quantile = clean.quantile(1.0 - alpha)
    tail = clean.loc[clean <= quantile]

    if tail.empty:
        return float("nan")

    return float(tail.mean())


def compute_downside_deviation(
    returns: pd.Series,
    *,
    annualization_periods: int = DEFAULT_ANNUALIZATION_PERIODS,
    threshold: float = 0.0,
) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if clean.empty:
        return float("nan")

    downside = np.minimum(clean.to_numpy(dtype=float) - threshold, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))

    return downside_deviation * float(np.sqrt(annualization_periods))


def compute_strategy_metrics(
    df: pd.DataFrame,
    *,
    return_col: str = "net_return_proxy",
    gross_return_col: str = "gross_return_proxy",
    eligible_col: str = "is_backtest_eligible",
    exposure_col: str = "target_exposure_for_backtest",
    delta_col: str = "delta_exposure",
    cost_col: str = "cost_proxy",
    annualization_periods: int = DEFAULT_ANNUALIZATION_PERIODS,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
) -> dict[str, Any]:
    """
    Compute Phase 10 performance metrics for one strategy panel.

    Metrics use only rows where is_backtest_eligible == True.
    Availability diagnostics use all rows.
    """
    if annualization_periods <= 0:
        raise BacktestMetricsError(
            f"annualization_periods must be positive. Got {annualization_periods}."
        )

    required = [return_col, eligible_col]
    _require_columns(df, required, "backtest panel")

    n_obs = int(len(df))
    eligible_mask = _as_bool_mask(df[eligible_col])
    n_eligible = int(eligible_mask.sum())
    availability_rate = float(n_eligible / n_obs) if n_obs > 0 else float("nan")

    result: dict[str, Any] = {
        "n_obs": n_obs,
        "n_eligible": n_eligible,
        "availability_rate": availability_rate,
    }

    if n_eligible == 0:
        result.update(
            {
                "start_date": None,
                "end_date": None,
                "n_return_obs": 0,
                "total_return_proxy": float("nan"),
                "mean_return": float("nan"),
                "median_return": float("nan"),
                "annualized_return": float("nan"),
                "annualized_volatility": float("nan"),
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "calmar": float("nan"),
                "max_drawdown": float("nan"),
                "hit_rate": float("nan"),
                "skew": float("nan"),
                "excess_kurtosis": float("nan"),
                "cte_95": float("nan"),
                "mean_exposure": float("nan"),
                "mean_abs_exposure": float("nan"),
                "sum_abs_exposure": float("nan"),
                "return_per_abs_exposure": float("nan"),
                "drawdown_per_abs_exposure": float("nan"),
                "return_to_drawdown": float("nan"),
                "turnover": float("nan"),
                "mean_abs_delta_exposure": float("nan"),
                "average_cost": float("nan"),
                "total_cost": float("nan"),
                "average_gross_return": float("nan"),
                "total_gross_return": float("nan"),
            }
        )
        result.update(get_metrics_metadata(horizon_trading_days=horizon_trading_days))
        return result

    eligible = df.loc[eligible_mask].copy()
    eligible[return_col] = pd.to_numeric(eligible[return_col], errors="coerce")

    if eligible[return_col].isna().any():
        n_missing = int(eligible[return_col].isna().sum())
        raise BacktestMetricsError(
            f"Eligible rows contain missing/non-numeric {return_col}. "
            f"Count={n_missing}."
        )

    returns = eligible[return_col].astype(float)
    n_return_obs = int(len(returns))

    if "target_trade_date" in eligible.columns:
        trade_dates = pd.to_datetime(eligible["target_trade_date"], errors="coerce")
        start_date = trade_dates.min()
        end_date = trade_dates.max()
    else:
        start_date = None
        end_date = None

    mean_return = float(returns.mean())
    median_return = float(returns.median())
    total_return_proxy = float(returns.sum())
    annualized_return = mean_return * float(annualization_periods)

    if n_return_obs >= 2:
        annualized_volatility = float(
            returns.std(ddof=1) * np.sqrt(float(annualization_periods))
        )
    else:
        annualized_volatility = float("nan")

    sharpe = _safe_ratio(annualized_return, annualized_volatility)

    downside_deviation = compute_downside_deviation(
        returns,
        annualization_periods=annualization_periods,
    )
    sortino = _safe_ratio(annualized_return, downside_deviation)

    max_drawdown = compute_max_drawdown(returns)
    calmar = _safe_ratio(annualized_return, abs(max_drawdown))

    hit_rate = float((returns > 0).mean())

    skew = float(returns.skew()) if n_return_obs >= 3 else np.nan
    excess_kurtosis = float(returns.kurt()) if n_return_obs >= 4 else np.nan
    cte_95 = compute_cte(returns, alpha=0.95)

    if exposure_col in eligible.columns:
        exposure = pd.to_numeric(eligible[exposure_col], errors="coerce")
        mean_exposure = float(exposure.mean())
        mean_abs_exposure = float(exposure.abs().mean())
        sum_abs_exposure = float(exposure.abs().sum())
    else:
        mean_exposure = float("nan")
        mean_abs_exposure = float("nan")
        sum_abs_exposure = float("nan")

    if delta_col in eligible.columns:
        delta = pd.to_numeric(eligible[delta_col], errors="coerce")
        turnover = float(delta.abs().sum())
        mean_abs_delta_exposure = float(delta.abs().mean())
    else:
        turnover = float("nan")
        mean_abs_delta_exposure = float("nan")

    if cost_col in eligible.columns:
        costs = pd.to_numeric(eligible[cost_col], errors="coerce")
        average_cost = float(costs.mean())
        total_cost = float(costs.sum())
    else:
        average_cost = float("nan")
        total_cost = float("nan")

    if gross_return_col in eligible.columns:
        gross_returns = pd.to_numeric(eligible[gross_return_col], errors="coerce")
        average_gross_return = float(gross_returns.mean())
        total_gross_return = float(gross_returns.sum())
    else:
        average_gross_return = float("nan")
        total_gross_return = float("nan")

    return_per_abs_exposure = _safe_ratio(total_return_proxy, sum_abs_exposure)
    drawdown_per_abs_exposure = _safe_ratio(max_drawdown, mean_abs_exposure)
    return_to_drawdown = _safe_ratio(total_return_proxy, abs(max_drawdown))

    result.update(
        {
            "start_date": None if pd.isna(start_date) else str(pd.Timestamp(start_date).date()),
            "end_date": None if pd.isna(end_date) else str(pd.Timestamp(end_date).date()),
            "n_return_obs": n_return_obs,
            "total_return_proxy": total_return_proxy,
            "mean_return": mean_return,
            "median_return": median_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_drawdown,
            "hit_rate": hit_rate,
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
            "cte_95": cte_95,
            "mean_exposure": mean_exposure,
            "mean_abs_exposure": mean_abs_exposure,
            "sum_abs_exposure": sum_abs_exposure,
            "return_per_abs_exposure": return_per_abs_exposure,
            "drawdown_per_abs_exposure": drawdown_per_abs_exposure,
            "return_to_drawdown": return_to_drawdown,
            "turnover": turnover,
            "mean_abs_delta_exposure": mean_abs_delta_exposure,
            "average_cost": average_cost,
            "total_cost": total_cost,
            "average_gross_return": average_gross_return,
            "total_gross_return": total_gross_return,
        }
    )

    result.update(get_metrics_metadata(horizon_trading_days=horizon_trading_days))

    return result


def build_strategy_metric_table(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("market", "strategy_name"),
    return_col: str = "net_return_proxy",
    annualization_periods: int = DEFAULT_ANNUALIZATION_PERIODS,
    horizon_trading_days: int = DEFAULT_HORIZON_TRADING_DAYS,
) -> pd.DataFrame:
    """
    Build one metrics row per market/strategy.

    Metrics are calculated only from eligible rows. n_obs and availability_rate
    still reflect all rows in each group.
    """
    _require_columns(panel, list(group_cols), "backtest panel")

    rows: list[dict[str, Any]] = []

    for group_key, group in panel.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            col: value
            for col, value in zip(group_cols, group_key, strict=True)
        }

        row.update(
            compute_strategy_metrics(
                group,
                return_col=return_col,
                annualization_periods=annualization_periods,
                horizon_trading_days=horizon_trading_days,
            )
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def build_availability_summary(
    panel: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("market", "strategy_name"),
    eligible_col: str = "is_backtest_eligible",
    exclusion_col: str = "exclusion_reason",
) -> pd.DataFrame:
    """
    Count exclusion reasons for each market/strategy.

    This uses all rows, not only eligible rows.
    """
    _require_columns(panel, [*group_cols, eligible_col, exclusion_col], "backtest panel")

    counts = (
        panel.groupby([*group_cols, exclusion_col], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )

    pivot = counts.pivot_table(
        index=list(group_cols),
        columns=exclusion_col,
        values="n_rows",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    pivot.columns.name = None

    total_rows = panel.groupby(list(group_cols), dropna=False).size().reset_index(name="n_obs")
    eligible_rows = (
        panel.loc[_as_bool_mask(panel[eligible_col])]
        .groupby(list(group_cols), dropna=False)
        .size()
        .reset_index(name="n_eligible")
    )

    out = total_rows.merge(eligible_rows, how="left", on=list(group_cols))
    out["n_eligible"] = out["n_eligible"].fillna(0).astype(int)
    out["availability_rate"] = out["n_eligible"] / out["n_obs"]

    out = out.merge(pivot, how="left", on=list(group_cols))

    reason_cols = [
        col for col in out.columns
        if col not in [*group_cols, "n_obs", "n_eligible", "availability_rate"]
    ]
    for col in reason_cols:
        out[col] = out[col].fillna(0).astype(int)

    return out.sort_values(list(group_cols)).reset_index(drop=True)