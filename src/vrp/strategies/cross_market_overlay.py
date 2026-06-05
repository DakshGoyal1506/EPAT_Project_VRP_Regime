from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from vrp.reports.cross_market import (
    CrossMarketInputError,
    CrossMarketLeakageError,
    _date64ns,
    _repo_path,
    _safe_to_csv,
    _safe_to_parquet,
    assert_no_same_date_us_leakage,
    guarded_read_parquet,
)


class CrossMarketOverlayError(RuntimeError):
    """Raised when Phase 13 cross-market overlay construction fails."""


def _coerce_date(df: pd.DataFrame, target_col: str = "date") -> pd.DataFrame:
    out = df.copy()

    if target_col not in out.columns:
        candidates = [
            "date",
            "Date",
            "datetime",
            "Datetime",
            "timestamp",
            "Timestamp",
            "trade_date",
            "session_date",
            "india_date",
            "signal_observation_date",
            "target_trade_date",
            "outcome_label_date",
            "signal_available_after_close_date",
        ]
        matched = [c for c in candidates if c in out.columns]
        if matched:
            out = out.rename(columns={matched[0]: target_col})
        elif isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            out = out.rename(columns={out.columns[0]: target_col})
        else:
            raise CrossMarketOverlayError(
                f"Could not find date column. Available columns: {list(out.columns)}"
            )

    out[target_col] = _date64ns(
        out[target_col],
        name=target_col,
    ).to_numpy(dtype="datetime64[ns]")
    if out[target_col].isna().any():
        n_bad = int(out[target_col].isna().sum())
        raise CrossMarketOverlayError(
            f"Date column {target_col!r} has {n_bad} invalid values."
        )

    return out.sort_values(target_col).drop_duplicates(target_col, keep="last")


def _first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = True,
    logical_name: str = "",
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise CrossMarketOverlayError(
            f"Missing required column for {logical_name or 'value'}. "
            f"Tried {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def _required_output_path(config: Mapping[str, Any], key: str) -> str:
    outputs = config.get("outputs", {})
    if key not in outputs:
        raise CrossMarketOverlayError(f"Missing outputs.{key} in config.")
    return str(outputs[key])


def _overlay_config(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(config.get("overlay", {}))
    cfg.setdefault("enabled", True)
    cfg.setdefault("analysis_only", True)
    cfg.setdefault("not_part_of_phase9_strategy_universe", True)
    cfg.setdefault("primary_india_strategy", "mar_prob_linear_carry")
    cfg.setdefault("secondary_india_strategy", "hmm_prob_linear_carry")
    cfg.setdefault("us_stress_cutoffs", [0.50, 0.60, 0.70])
    cfg.setdefault("default_us_stress_cutoff", 0.60)
    cfg.setdefault("cost_bps", 5)
    cfg.setdefault("no_phase9_mutation", True)
    cfg.setdefault("no_phase10_mutation", True)
    cfg.setdefault("no_phase11_usage", True)
    return cfg


def _strategy_name_candidates(strategy_name: str) -> list[str]:
    base = str(strategy_name)
    return [
        base,
        base.lower(),
        base.upper(),
        base.replace("_", "-"),
        base.replace("-", "_"),
    ]


def _filter_strategy_rows(
    df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    strategy_col = _first_existing_column(
        df,
        [
            "strategy",
            "strategy_name",
            "signal_name",
            "rule_name",
            "model_strategy",
            "strategy_id",
        ],
        required=False,
        logical_name="strategy name",
    )

    if strategy_col is None:
        return df.copy()

    allowed = set(_strategy_name_candidates(strategy_name))
    mask = df[strategy_col].astype(str).isin(allowed)

    if not mask.any():
        unique_values = sorted(df[strategy_col].astype(str).dropna().unique().tolist())
        raise CrossMarketOverlayError(
            f"Strategy {strategy_name!r} not found in strategy signals. "
            f"Column={strategy_col!r}. Available values={unique_values[:30]}"
        )

    return df.loc[mask].copy()


def load_india_strategy_signals(
    path: str | Path,
    strategy_name: str,
    config: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load Phase 9 India strategy signals for one base strategy.

    This function reads but never modifies Phase 9 signal files.
    """
    if config is not None:
        df = guarded_read_parquet(path, config=config, root=root)
    else:
        abs_path = _repo_path(path, root)
        if not abs_path.exists():
            raise CrossMarketOverlayError(f"Strategy signal file not found: {abs_path}")
        df = pd.read_parquet(abs_path)

    filtered = _filter_strategy_rows(df, strategy_name)
    filtered = _coerce_date(filtered, target_col="date")
    filtered["strategy"] = strategy_name

    return filtered.sort_values("date").reset_index(drop=True)


def load_india_backtest_panel(
    path: str | Path,
    config: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load locked Phase 10 India backtest panel.

    This function reads but never modifies Phase 10 backtest files.
    """
    if config is not None:
        df = guarded_read_parquet(path, config=config, root=root)
    else:
        abs_path = _repo_path(path, root)
        if not abs_path.exists():
            raise CrossMarketOverlayError(f"Backtest panel file not found: {abs_path}")
        df = pd.read_parquet(abs_path)

    out = _coerce_date(df, target_col="date")
    return out.sort_values("date").reset_index(drop=True)


def select_base_strategy_exposure(
    strategy_df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """
    Select the Phase 9 base exposure series for the requested India strategy.

    Supports common signal formats:
    - long format: date, strategy, exposure
    - wide format: date, <strategy_name>_exposure
    - generic format: date, target_exposure
    """
    df = _filter_strategy_rows(strategy_df, strategy_name)
    df = _coerce_date(df, target_col="date")

    exact_candidates = [
        f"{strategy_name}_exposure",
        f"{strategy_name}_target_exposure",
        f"{strategy_name}_signal_exposure",
    ]

    generic_candidates = [
        "target_exposure",
        "exposure",
        "signal_exposure",
        "position",
        "target_position",
        "weight",
        "signal",
    ]

    exposure_col = _first_existing_column(
        df,
        exact_candidates + generic_candidates,
        required=True,
        logical_name="base strategy exposure",
    )

    out = pd.DataFrame(
        {
            "date": df["date"],
            "strategy": strategy_name,
            "base_exposure": pd.to_numeric(df[exposure_col], errors="coerce"),
        }
    )

    if out["base_exposure"].isna().all():
        raise CrossMarketOverlayError(
            f"Exposure column {exposure_col!r} has no numeric observations."
        )

    out = out.dropna(subset=["base_exposure"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _select_backtest_return_column(
    backtest_df: pd.DataFrame,
    strategy_name: str,
) -> str:
    """
    Infer base return/payoff column from Phase 10 backtest panel.
    """
    candidates = [
        f"{strategy_name}_net_return",
        f"{strategy_name}_return",
        f"{strategy_name}_pnl",
        f"{strategy_name}_payoff",
        "strategy_net_return",
        "strategy_return",
        "net_return_proxy",
        "gross_return_proxy",
        "net_return",
        "return",
        "daily_return",
        "pnl",
        "daily_pnl",
        "payoff",
        "forward_vrp_payoff",
        "phase10_forward_vrp_research_proxy",
    ]

    return cast(
        str,
        _first_existing_column(
            backtest_df,
            candidates,
            required=True,
            logical_name="Phase 10 base return/payoff proxy",
        ),
    )


def _filter_backtest_strategy_rows(
    backtest_df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    strategy_col = _first_existing_column(
        backtest_df,
        [
            "strategy",
            "strategy_name",
            "signal_name",
            "rule_name",
            "model_strategy",
            "strategy_id",
        ],
        required=False,
        logical_name="backtest strategy name",
    )

    if strategy_col is None:
        return backtest_df.copy()

    allowed = set(_strategy_name_candidates(strategy_name))
    mask = backtest_df[strategy_col].astype(str).isin(allowed)

    if not mask.any():
        unique_values = sorted(
            backtest_df[strategy_col].astype(str).dropna().unique().tolist()
        )
        raise CrossMarketOverlayError(
            f"Strategy {strategy_name!r} not found in backtest panel. "
            f"Column={strategy_col!r}. Available values={unique_values[:30]}"
        )

    return backtest_df.loc[mask].copy()


def select_base_strategy_returns(
    backtest_df: pd.DataFrame,
    strategy_name: str,
) -> pd.DataFrame:
    """
    Select base Phase 10 return/payoff proxy for the requested strategy.
    """
    df = _filter_backtest_strategy_rows(backtest_df, strategy_name)
    df = _coerce_date(df, target_col="date")

    return_col = _select_backtest_return_column(df, strategy_name)

    out = pd.DataFrame(
        {
            "date": df["date"],
            "strategy": strategy_name,
            "base_return": pd.to_numeric(df[return_col], errors="coerce"),
        }
    )

    if out["base_return"].isna().all():
        raise CrossMarketOverlayError(
            f"Return column {return_col!r} has no numeric observations."
        )

    out = out.dropna(subset=["base_return"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def _select_predictive_overlay_columns(
    predictive_panel: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    required = {"india_date", "us_lagged_date", "lag_calendar_days"}
    missing = sorted(required - set(predictive_panel.columns))
    if missing:
        raise CrossMarketOverlayError(
            f"Predictive panel missing required overlay columns: {missing}"
        )

    if "model" in predictive_panel.columns:
        model_panel = predictive_panel[
            predictive_panel["model"].astype(str) == str(model)
        ]
        if model_panel.empty:
            available = sorted(
                predictive_panel["model"].astype(str).dropna().unique().tolist()
            )
            raise CrossMarketOverlayError(
                f"Model {model!r} not found in predictive panel. Available={available}"
            )
    else:
        model_panel = predictive_panel.copy()

    stress_col = _first_existing_column(
        model_panel,
        ["us_stress_prob_lag1", "us_stress_prob"],
        required=True,
        logical_name="lagged US stress probability",
    )

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(model_panel["india_date"]).dt.normalize(),
            "india_date": pd.to_datetime(model_panel["india_date"]).dt.normalize(),
            "us_lagged_date": pd.to_datetime(
                model_panel["us_lagged_date"]
            ).dt.normalize(),
            "lag_calendar_days": pd.to_numeric(
                model_panel["lag_calendar_days"],
                errors="coerce",
            ),
            "us_stress_prob_lagged": pd.to_numeric(
                model_panel[stress_col],
                errors="coerce",
            ),
        }
    )

    if "lag_is_strictly_prior" in model_panel.columns:
        out["lag_is_strictly_prior"] = model_panel["lag_is_strictly_prior"].astype(bool)
    else:
        out["lag_is_strictly_prior"] = out["us_lagged_date"] < out["india_date"]

    leakage_check = out.rename(columns={"date": "_date_unused"})
    assert_no_same_date_us_leakage(
        leakage_check[["india_date", "us_lagged_date", "lag_calendar_days"]].copy()
    )

    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def apply_us_stress_overlay_to_exposure(
    exposure_df: pd.DataFrame,
    predictive_panel: pd.DataFrame,
    *,
    model: str,
    cutoff: float,
) -> pd.DataFrame:
    """
    Apply the Phase 13 analysis-only overlay rule:

        if lagged_US_stress_prob > cutoff:
            overlay_exposure = 0.0
        else:
            overlay_exposure = base_exposure

    Uses only lagged US information where us_lagged_date < india_date.
    """
    base = exposure_df.copy()
    base = _coerce_date(base, target_col="date")

    if "base_exposure" not in base.columns:
        raise CrossMarketOverlayError("exposure_df must contain base_exposure.")

    overlay_features = _select_predictive_overlay_columns(
        predictive_panel,
        model=model,
    )

    panel = base.merge(overlay_features, on="date", how="inner")

    if panel.empty:
        raise CrossMarketOverlayError(
            "Overlay merge produced no rows. Check Phase 9 signal dates and "
            "Phase 13 predictive panel dates."
        )

    panel["cutoff"] = float(cutoff)
    panel["blocked_by_us_stress"] = panel["us_stress_prob_lagged"] > float(cutoff)
    panel["overlay_exposure"] = np.where(
        panel["blocked_by_us_stress"],
        0.0,
        pd.to_numeric(panel["base_exposure"], errors="coerce"),
    )

    panel["analysis_only"] = True
    panel["not_part_of_phase9_strategy_universe"] = True
    panel["phase9_mutation"] = False
    panel["phase10_mutation"] = False
    panel["phase11_usage"] = False

    assert_no_same_date_us_leakage(panel)

    return panel.sort_values("date").reset_index(drop=True)


def compute_overlay_returns(
    overlay_panel: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Compute analysis-only overlay return proxy.

    The Phase 10 base return is scaled by the exposure ratio:

        overlay_return_gross = base_return * overlay_exposure / base_exposure

    If base_exposure is zero, overlay return is set to zero to avoid division
    blow-ups.

    Transaction-cost proxy:
        cost_bps * abs(delta overlay_exposure) / 10000
    """
    panel = overlay_panel.copy()
    panel = _coerce_date(panel, target_col="date")

    rets = returns_df.copy()
    rets = _coerce_date(rets, target_col="date")

    if "base_return" not in rets.columns:
        raise CrossMarketOverlayError("returns_df must contain base_return.")

    out = panel.merge(
        rets[["date", "strategy", "base_return"]],
        on=["date", "strategy"],
        how="inner",
    )

    if out.empty:
        raise CrossMarketOverlayError(
            "Overlay return merge produced no rows. Check backtest panel dates."
        )

    base_exposure = pd.to_numeric(out["base_exposure"], errors="coerce")
    overlay_exposure = pd.to_numeric(out["overlay_exposure"], errors="coerce")
    base_return = pd.to_numeric(out["base_return"], errors="coerce")

    exposure_ratio = pd.Series(0.0, index=out.index)
    nonzero_base = base_exposure.abs() > 1e-12
    exposure_ratio.loc[nonzero_base] = (
        overlay_exposure.loc[nonzero_base] / base_exposure.loc[nonzero_base]
    )

    out["overlay_return_gross"] = base_return * exposure_ratio
    out["base_turnover"] = base_exposure.diff().abs().fillna(base_exposure.abs())
    out["overlay_turnover"] = overlay_exposure.diff().abs().fillna(
        overlay_exposure.abs()
    )

    out["overlay_cost"] = (float(cost_bps) / 10000.0) * out["overlay_turnover"]
    out["overlay_return"] = out["overlay_return_gross"] - out["overlay_cost"]

    out["base_equity"] = (1.0 + out["base_return"].fillna(0.0)).cumprod()
    out["overlay_equity"] = (1.0 + out["overlay_return"].fillna(0.0)).cumprod()

    return out.sort_values("date").reset_index(drop=True)


def _annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    vol = float(r.std(ddof=1))
    if vol <= 0.0 or not np.isfinite(vol):
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / vol)


def _annualized_sortino(returns: pd.Series, periods: int = 252) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0.0]
    if len(downside) < 2:
        return float("nan")
    downside_vol = float(downside.std(ddof=1))
    if downside_vol <= 0.0 or not np.isfinite(downside_vol):
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / downside_vol)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def summarize_overlay_vs_base(
    overlay_return_panel: pd.DataFrame,
    *,
    model: str,
    strategy: str,
    cutoff: float,
) -> pd.DataFrame:
    """
    Summarize base versus analysis-only overlay results.
    """
    df = overlay_return_panel.copy()

    required = {
        "base_return",
        "overlay_return",
        "base_exposure",
        "overlay_exposure",
        "base_turnover",
        "overlay_turnover",
        "blocked_by_us_stress",
        "analysis_only",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise CrossMarketOverlayError(f"Cannot summarize overlay. Missing: {missing}")

    n_obs = int(len(df))
    if n_obs == 0:
        raise CrossMarketOverlayError("Cannot summarize empty overlay panel.")

    summary = {
        "model": model,
        "strategy": strategy,
        "cutoff": float(cutoff),
        "n_obs": n_obs,
        "base_mean_return": float(
            pd.to_numeric(df["base_return"], errors="coerce").mean()
        ),
        "overlay_mean_return": float(
            pd.to_numeric(df["overlay_return"], errors="coerce").mean()
        ),
        "base_vol": float(
            pd.to_numeric(df["base_return"], errors="coerce").std(ddof=1)
        ),
        "overlay_vol": float(
            pd.to_numeric(df["overlay_return"], errors="coerce").std(ddof=1)
        ),
        "base_sharpe": _annualized_sharpe(df["base_return"]),
        "overlay_sharpe": _annualized_sharpe(df["overlay_return"]),
        "base_sortino": _annualized_sortino(df["base_return"]),
        "overlay_sortino": _annualized_sortino(df["overlay_return"]),
        "base_max_drawdown": _max_drawdown_from_returns(df["base_return"]),
        "overlay_max_drawdown": _max_drawdown_from_returns(df["overlay_return"]),
        "base_turnover": float(
            pd.to_numeric(df["base_turnover"], errors="coerce").sum()
        ),
        "overlay_turnover": float(
            pd.to_numeric(df["overlay_turnover"], errors="coerce").sum()
        ),
        "base_exposure_mean": float(
            pd.to_numeric(df["base_exposure"], errors="coerce").mean()
        ),
        "overlay_exposure_mean": float(
            pd.to_numeric(df["overlay_exposure"], errors="coerce").mean()
        ),
        "blocked_days": int(df["blocked_by_us_stress"].fillna(False).sum()),
        "blocked_day_fraction": float(df["blocked_by_us_stress"].fillna(False).mean()),
        "analysis_only": bool(df["analysis_only"].all()),
    }

    return pd.DataFrame([summary])


def validate_overlay_analysis_only(
    overlay_panel: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """
    Validate Phase 13 overlay contract.

    Fail if overlay is not explicitly analysis-only or if no-leakage rules fail.
    """
    overlay_cfg = _overlay_config(config)

    if overlay_cfg.get("analysis_only") is not True:
        raise CrossMarketOverlayError("Config overlay.analysis_only must be true.")

    if overlay_cfg.get("no_phase9_mutation") is not True:
        raise CrossMarketOverlayError(
            "Config overlay.no_phase9_mutation must be true."
        )

    if overlay_cfg.get("no_phase10_mutation") is not True:
        raise CrossMarketOverlayError(
            "Config overlay.no_phase10_mutation must be true."
        )

    if overlay_cfg.get("no_phase11_usage") is not True:
        raise CrossMarketOverlayError(
            "Config overlay.no_phase11_usage must be true."
        )

    required_flags = {
        "analysis_only": True,
        "not_part_of_phase9_strategy_universe": True,
        "phase9_mutation": False,
        "phase10_mutation": False,
        "phase11_usage": False,
    }

    for col, expected in required_flags.items():
        if col not in overlay_panel.columns:
            raise CrossMarketOverlayError(f"Overlay panel missing required flag: {col}")
        values = overlay_panel[col].dropna().unique().tolist()
        if not all(bool(v) == bool(expected) for v in values):
            raise CrossMarketOverlayError(
                f"Overlay flag {col!r} violates expected value {expected!r}. "
                f"Observed={values}"
            )

    assert_no_same_date_us_leakage(overlay_panel)


def build_india_cross_market_overlay_panel(
    predictive_panel: pd.DataFrame,
    strategy_signals: pd.DataFrame,
    backtest_panel: pd.DataFrame,
    *,
    model: str,
    strategy_name: str,
    cutoff: float,
    cost_bps: float = 5.0,
    config: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one analysis-only India cross-market overlay panel and summary.
    """
    exposure = select_base_strategy_exposure(
        strategy_signals,
        strategy_name=strategy_name,
    )
    returns = select_base_strategy_returns(
        backtest_panel,
        strategy_name=strategy_name,
    )

    overlay_exposure_panel = apply_us_stress_overlay_to_exposure(
        exposure,
        predictive_panel,
        model=model,
        cutoff=float(cutoff),
    )

    overlay_return_panel = compute_overlay_returns(
        overlay_exposure_panel,
        returns,
        cost_bps=float(cost_bps),
    )

    overlay_return_panel["model"] = model
    overlay_return_panel["strategy"] = strategy_name
    overlay_return_panel["cutoff"] = float(cutoff)

    if config is not None:
        validate_overlay_analysis_only(overlay_return_panel, config=config)

    summary = summarize_overlay_vs_base(
        overlay_return_panel,
        model=model,
        strategy=strategy_name,
        cutoff=float(cutoff),
    )

    return overlay_return_panel, summary


def _strategy_for_model(
    model: str,
    config: Mapping[str, Any],
) -> str:
    overlay_cfg = _overlay_config(config)

    if model == "markov_autoreg":
        return str(overlay_cfg.get("primary_india_strategy", "mar_prob_linear_carry"))

    if model == "gaussian_hmm":
        return str(overlay_cfg.get("secondary_india_strategy", "hmm_prob_linear_carry"))

    # Conservative fallback.
    return str(overlay_cfg.get("primary_india_strategy", "mar_prob_linear_carry"))


def build_all_india_cross_market_overlays(
    predictive_panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    models: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build Phase 13 analysis-only India overlays for all requested models/cutoffs.

    This function reads locked Phase 9/10 inputs but does not modify them.
    """
    overlay_cfg = _overlay_config(config)
    if overlay_cfg.get("enabled") is not True:
        return {
            "india_overlay_panel": pd.DataFrame(),
            "overlay_summary": pd.DataFrame(),
        }

    inputs = config.get("input_files", {})
    india_signal_path = inputs.get("INDIA", {}).get("strategy_signals")
    india_backtest_path = inputs.get("INDIA", {}).get("backtest")

    if not india_signal_path or not india_backtest_path:
        raise CrossMarketOverlayError(
            "Missing India strategy_signals/backtest input paths in config."
        )

    requested_models = models or list(config.get("models", []))
    cutoffs = overlay_cfg.get("us_stress_cutoffs", [0.50, 0.60, 0.70])

    if not requested_models:
        raise CrossMarketOverlayError("No models requested for overlay build.")

    strategy_signals_cache: dict[str, pd.DataFrame] = {}
    backtest_cache: dict[str, pd.DataFrame] = {}

    overlay_parts: list[pd.DataFrame] = []
    summary_parts: list[pd.DataFrame] = []

    for model in requested_models:
        strategy_name = _strategy_for_model(str(model), config)

        if strategy_name not in strategy_signals_cache:
            strategy_signals_cache[strategy_name] = load_india_strategy_signals(
                india_signal_path,
                strategy_name=strategy_name,
                config=config,
                root=root,
            )

        if strategy_name not in backtest_cache:
            raw_backtest = load_india_backtest_panel(
                india_backtest_path,
                config=config,
                root=root,
            )
            backtest_cache[strategy_name] = _filter_backtest_strategy_rows(
                raw_backtest,
                strategy_name=strategy_name,
            )

        for cutoff in cutoffs:
            overlay_panel, summary = build_india_cross_market_overlay_panel(
                predictive_panel=predictive_panel,
                strategy_signals=strategy_signals_cache[strategy_name],
                backtest_panel=backtest_cache[strategy_name],
                model=str(model),
                strategy_name=strategy_name,
                cutoff=float(cutoff),
                cost_bps=float(overlay_cfg.get("cost_bps", 5.0)),
                config=config,
            )
            overlay_parts.append(overlay_panel)
            summary_parts.append(summary)

    all_overlay = (
        pd.concat(overlay_parts, axis=0, ignore_index=True)
        if overlay_parts
        else pd.DataFrame()
    )
    all_summary = (
        pd.concat(summary_parts, axis=0, ignore_index=True)
        if summary_parts
        else pd.DataFrame()
    )

    return {
        "india_overlay_panel": all_overlay,
        "overlay_summary": all_summary,
    }


def write_india_cross_market_overlay_outputs(
    overlay_outputs: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    root: str | Path | None = None,
) -> dict[str, str]:
    """
    Write Phase 13 overlay panel and summary outputs.
    """
    written: dict[str, str] = {}

    if "india_overlay_panel" in overlay_outputs:
        panel = overlay_outputs["india_overlay_panel"]
        path = _required_output_path(config, "india_overlay_panel")
        written["india_overlay_panel"] = str(_safe_to_parquet(panel, path, root=root))

    if "overlay_summary" in overlay_outputs:
        summary = overlay_outputs["overlay_summary"]
        path = _required_output_path(config, "overlay_summary")
        written["overlay_summary"] = str(_safe_to_csv(summary, path, root=root))

    return written


def run_india_cross_market_overlay(
    predictive_panel: pd.DataFrame,
    config: Mapping[str, Any],
    root: str | Path | None = None,
    *,
    models: list[str] | None = None,
    write_outputs: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Build and optionally write all Phase 13 India cross-market overlays.
    """
    overlay_outputs = build_all_india_cross_market_overlays(
        predictive_panel=predictive_panel,
        config=config,
        root=root,
        models=models,
    )

    if write_outputs:
        written = write_india_cross_market_overlay_outputs(
            overlay_outputs,
            config=config,
            root=root,
        )
        overlay_outputs["written_outputs"] = pd.DataFrame(
            [{"output": key, "path": value} for key, value in written.items()]
        )

    return overlay_outputs


def overlay_summary_required_columns(config: Mapping[str, Any]) -> list[str]:
    cols = config.get("required_overlay_summary_columns", [])
    if isinstance(cols, list) and cols:
        return [str(c) for c in cols]

    return [
        "model",
        "strategy",
        "cutoff",
        "n_obs",
        "base_mean_return",
        "overlay_mean_return",
        "base_vol",
        "overlay_vol",
        "base_sharpe",
        "overlay_sharpe",
        "base_sortino",
        "overlay_sortino",
        "base_max_drawdown",
        "overlay_max_drawdown",
        "base_turnover",
        "overlay_turnover",
        "base_exposure_mean",
        "overlay_exposure_mean",
        "blocked_days",
        "blocked_day_fraction",
        "analysis_only",
    ]


def validate_overlay_summary_schema(
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    required = set(overlay_summary_required_columns(config))
    missing = sorted(required - set(summary.columns))
    if missing:
        raise CrossMarketOverlayError(
            f"Overlay summary missing required columns: {missing}"
        )

    if "analysis_only" in summary.columns:
        if not summary["analysis_only"].fillna(False).astype(bool).all():
            raise CrossMarketOverlayError("Overlay summary has analysis_only != true.")

    if "cutoff" in summary.columns:
        bad_cutoffs = summary["cutoff"].isna()
        if bad_cutoffs.any():
            raise CrossMarketOverlayError("Overlay summary has missing cutoff values.")

    if "blocked_day_fraction" in summary.columns:
        frac = pd.to_numeric(summary["blocked_day_fraction"], errors="coerce")
        bad = (frac < 0.0) | (frac > 1.0)
        if bad.any():
            raise CrossMarketOverlayError(
                "Overlay summary blocked_day_fraction outside [0, 1]."
            )


def export_overlay_contract_json(
    config: Mapping[str, Any],
    path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """
    Optional helper to write a compact overlay contract JSON for report review.
    """
    out_path = _repo_path(path, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "overlay": config.get("overlay", {}),
        "required_overlay_summary_columns": overlay_summary_required_columns(config),
        "no_leakage_acceptance_rules": config.get("no_leakage_acceptance_rules", []),
        "interpretation": (
            "Phase 13 overlay is analysis-only. It is not part of the locked "
            "Phase 9 strategy universe and must not mutate Phase 10 backtest outputs."
        ),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    return out_path
