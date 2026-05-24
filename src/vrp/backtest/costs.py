from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class BacktestCostError(ValueError):
    """Raised when Phase 10 cost accounting is invalid."""


def validate_cost_bps(cost_bps: float) -> float:
    try:
        parsed = float(cost_bps)
    except Exception as exc:
        raise BacktestCostError(f"cost_bps must be numeric. Got {cost_bps!r}.") from exc

    if parsed < 0:
        raise BacktestCostError(f"cost_bps must be non-negative. Got {parsed}.")

    return parsed


def compute_exposure_change_costs(
    df: pd.DataFrame,
    *,
    exposure_col: str = "target_exposure_for_backtest",
    eligible_col: str = "is_backtest_eligible",
    group_cols: Sequence[str] = ("market", "strategy_name"),
    sort_cols: Sequence[str] = ("target_trade_date", "signal_observation_date"),
    cost_bps: float = 5.0,
    delta_col: str = "delta_exposure",
    cost_col: str = "cost_proxy",
) -> pd.DataFrame:
    """
    Compute exposure-change costs across eligible rows only.

    Rule:
        Within each market/strategy group, sort by target_trade_date and
        signal_observation_date. Initial exposure before first eligible row is 0.0.

        delta_exposure = current_exposure - previous_eligible_exposure
        cost_proxy = abs(delta_exposure) * cost_bps / 10000

    Ineligible rows get NaN delta/cost. This prevents silent cost accounting
    across unavailable strategy rows or missing outcome periods.
    """
    cost_bps = validate_cost_bps(cost_bps)

    out = df.copy()

    required = [exposure_col, eligible_col, *group_cols, *sort_cols]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise BacktestCostError(f"Missing required columns for cost accounting: {missing}")

    out[delta_col] = np.nan
    out[cost_col] = np.nan

    eligible_mask = out[eligible_col].fillna(False).astype(bool)
    if not bool(eligible_mask.any()):
        return out

    eligible = out.loc[eligible_mask].copy()
    eligible[exposure_col] = pd.to_numeric(eligible[exposure_col], errors="coerce")

    if eligible[exposure_col].isna().any():
        bad_count = int(eligible[exposure_col].isna().sum())
        raise BacktestCostError(
            f"Eligible rows contain non-finite {exposure_col}. Count={bad_count}."
        )

    for col in sort_cols:
        eligible[col] = pd.to_datetime(eligible[col], errors="coerce")

    eligible = eligible.sort_values([*group_cols, *sort_cols])

    for _, group in eligible.groupby(list(group_cols), sort=False):
        current_exposure = group[exposure_col].astype(float)
        previous_exposure = current_exposure.shift(1)
        previous_exposure.iloc[0] = 0.0

        delta = current_exposure - previous_exposure
        cost = delta.abs() * cost_bps / 10000.0

        out.loc[group.index, delta_col] = delta.to_numpy(dtype=float)
        out.loc[group.index, cost_col] = cost.to_numpy(dtype=float)

    return out


def apply_costs_to_backtest_panel(
    df: pd.DataFrame,
    *,
    gross_return_col: str = "gross_return_proxy",
    cost_col: str = "cost_proxy",
    net_return_col: str = "net_return_proxy",
    enabled: bool = True,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Add delta_exposure, cost_proxy, and net_return_proxy.

    If costs are disabled:
        cost_proxy = 0.0 on eligible rows
        net_return_proxy = gross_return_proxy on eligible rows

    Ineligible rows keep NaN net returns.
    """
    out = df.copy()

    if "is_backtest_eligible" not in out.columns:
        raise BacktestCostError("Missing is_backtest_eligible column.")

    if gross_return_col not in out.columns:
        raise BacktestCostError(f"Missing {gross_return_col} column.")

    eligible_mask = out["is_backtest_eligible"].fillna(False).astype(bool)

    if enabled:
        out = compute_exposure_change_costs(
            out,
            cost_bps=cost_bps,
            cost_col=cost_col,
        )
    else:
        out["delta_exposure"] = np.nan
        out[cost_col] = np.nan
        out.loc[eligible_mask, "delta_exposure"] = 0.0
        out.loc[eligible_mask, cost_col] = 0.0

    out[net_return_col] = np.nan
    out.loc[eligible_mask, net_return_col] = (
        pd.to_numeric(out.loc[eligible_mask, gross_return_col], errors="coerce")
        - pd.to_numeric(out.loc[eligible_mask, cost_col], errors="coerce").fillna(0.0)
    )

    return out