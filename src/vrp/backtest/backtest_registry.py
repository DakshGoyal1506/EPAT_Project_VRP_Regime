from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


BACKTEST_STRATEGY_UNIVERSE: tuple[str, ...] = (
    "unconditional_full",
    "threshold_hard_filter",
    "threshold_defensive",
    "hmm_prob_linear",
    "hmm_prob_linear_carry",
    "mar_prob_linear",
    "mar_prob_linear_carry",
)

BACKTEST_FORBIDDEN_SIGNAL_COLUMNS: tuple[str, ...] = (
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
)

BACKTEST_ALLOWED_OUTCOME_LABELS: tuple[str, ...] = (
    "vrp_forward_expost_gk_label",
)

BACKTEST_FORBIDDEN_STRATEGY_SUBSTRINGS: tuple[str, ...] = (
    "msvol",
)

BACKTEST_FORBIDDEN_SMOOTHED_PROBABILITY_SUBSTRINGS: tuple[str, ...] = (
    "smoothed",
    "smooth_prob",
    "smoother",
    "full_sample",
    "backward_prob",
    "forward_backward",
)

BACKTEST_ALLOWED_OUTCOME_ALIGNMENTS: tuple[str, ...] = (
    "signal_observation_date",
    "target_trade_date",
)


class BacktestRegistryError(ValueError):
    """Raised when Phase 10 registry rules are violated."""


def _as_string_set(values: Iterable[str]) -> set[str]:
    return {str(value) for value in values}


def assert_strategy_universe_locked(strategies: Iterable[str]) -> None:
    observed = _as_string_set(strategies)
    expected = set(BACKTEST_STRATEGY_UNIVERSE)

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing or extra:
        raise BacktestRegistryError(
            "Phase 10 strategy universe is locked and must match Phase 9 exactly. "
            f"Missing={missing}; Extra={extra}"
        )


def assert_no_outcome_labels_used_as_signals(signal_cols: Iterable[str]) -> None:
    cols = _as_string_set(signal_cols)
    forbidden = sorted(cols.intersection(BACKTEST_FORBIDDEN_SIGNAL_COLUMNS))

    if forbidden:
        raise BacktestRegistryError(
            "Forward/ex-post outcome labels are forbidden as strategy signal columns "
            f"in Phase 10. Forbidden columns found: {forbidden}. "
            "They are allowed only as realised outcome labels after the signal panel "
            "has already been created."
        )


def assert_payoff_label_is_outcome_only(label_col: str) -> None:
    if label_col not in BACKTEST_ALLOWED_OUTCOME_LABELS:
        raise BacktestRegistryError(
            f"Unsupported Phase 10 payoff label: {label_col!r}. "
            f"Allowed outcome labels: {list(BACKTEST_ALLOWED_OUTCOME_LABELS)}"
        )

    if label_col in BACKTEST_FORBIDDEN_SIGNAL_COLUMNS:
        return

    raise BacktestRegistryError(
        f"Payoff label {label_col!r} is not registered as forbidden for signal use. "
        "Every ex-post payoff label must be forbidden as a signal feature."
    )


def assert_no_msvol_strategy_use(df: pd.DataFrame) -> None:
    if "strategy_name" not in df.columns:
        return

    strategy_series = df["strategy_name"].astype(str).str.lower()
    mask = pd.Series(False, index=df.index)

    for token in BACKTEST_FORBIDDEN_STRATEGY_SUBSTRINGS:
        mask = mask | strategy_series.str.contains(token, na=False)

    n_bad = int(mask.sum())
    if n_bad > 0:
        bad_names = sorted(df.loc[mask, "strategy_name"].astype(str).unique().tolist())
        raise BacktestRegistryError(
            "MSVOL strategies are appendix-only diagnostics and are forbidden in "
            f"Phase 10 strategy backtests. Found {n_bad} rows: {bad_names}"
        )


def assert_no_smoothed_probability_use(df: pd.DataFrame) -> None:
    bad_columns: list[str] = []

    for col in df.columns:
        lower_col = str(col).lower()
        if any(token in lower_col for token in BACKTEST_FORBIDDEN_SMOOTHED_PROBABILITY_SUBSTRINGS):
            bad_columns.append(str(col))

    if bad_columns:
        raise BacktestRegistryError(
            "Full-sample smoothed probabilities are forbidden for Phase 10 backtests. "
            f"Forbidden probability columns found: {sorted(bad_columns)}"
        )

    if "probability_source" in df.columns:
        source_series = df["probability_source"].astype(str).str.lower()
        mask = pd.Series(False, index=df.index)

        for token in BACKTEST_FORBIDDEN_SMOOTHED_PROBABILITY_SUBSTRINGS:
            mask = mask | source_series.str.contains(token, na=False)

        if int(mask.sum()) > 0:
            bad_sources = sorted(df.loc[mask, "probability_source"].astype(str).unique().tolist())
            raise BacktestRegistryError(
                "Smoothed probability sources are forbidden in Phase 10. "
                f"Found sources: {bad_sources}"
            )


def assert_outcome_alignment_allowed(alignment: str) -> None:
    if alignment not in BACKTEST_ALLOWED_OUTCOME_ALIGNMENTS:
        raise BacktestRegistryError(
            f"Unsupported outcome alignment: {alignment!r}. "
            f"Allowed alignments: {list(BACKTEST_ALLOWED_OUTCOME_ALIGNMENTS)}"
        )