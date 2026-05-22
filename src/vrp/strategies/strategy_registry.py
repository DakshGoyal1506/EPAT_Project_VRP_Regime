from __future__ import annotations

from collections.abc import Iterable, Mapping


APPROVED_STRATEGY_NAMES: tuple[str, ...] = (
    "unconditional_full",
    "threshold_hard_filter",
    "threshold_defensive",
    "hmm_prob_linear",
    "hmm_prob_linear_carry",
    "mar_prob_linear",
    "mar_prob_linear_carry",
)

REJECTED_STRATEGY_NAMES: tuple[str, ...] = (
    "threshold_carry_aware",
    "hmm_hard",
    "hmm_defensive",
    "mar_hard",
    "mar_defensive",
    "probability_product",
    "probability_cutoff_filter",
    "msvol",
    "msgarch",
)

ALLOWED_STRATEGY_MODELS: tuple[str, ...] = (
    "unconditional",
    "threshold",
    "gaussian_hmm",
    "markov_autoreg",
)

FORBIDDEN_STRATEGY_MODELS: tuple[str, ...] = (
    "msvol",
    "msgarch",
)

STRATEGY_FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "future",
    "expost",
    "ex_post",
    "smoothed",
    "diagnostic",
    "crisis",
    "msvol",
    "msgarch",
)

STRATEGY_FORBIDDEN_EXACT_COLUMNS: tuple[str, ...] = (
    "rv_gk_22d_forward_ann_label",
    "vrp_forward_expost_gk_label",
)

STRATEGY_FORBIDDEN_SUFFIXES: tuple[str, ...] = (
    "_label",
)

ALLOWED_NEXT_SESSION_SUBSTRINGS: tuple[str, ...] = (
    "next_session",
)

ALLOWED_HMM_PROBABILITY_COLUMNS: tuple[str, ...] = (
    "hmm_filtered_prob_calm_for_next_session",
    "hmm_filtered_prob_transition_for_next_session",
    "hmm_filtered_prob_stress_for_next_session",
)

ALLOWED_MAR_PROBABILITY_COLUMNS: tuple[str, ...] = (
    "mar_filtered_prob_calm_for_next_session",
    "mar_filtered_prob_transition_for_next_session",
    "mar_filtered_prob_stress_for_next_session",
)

ALLOWED_CARRY_COLUMNS: tuple[str, ...] = (
    "vrp_har_gk",
    "vrp_har_gk_positive",
    "har_forecast_available",
)

ALLOWED_THRESHOLD_COLUMNS: tuple[str, ...] = (
    "threshold_state",
    "threshold_state_name",
    "threshold_regime_available",
    "threshold_trigger_reason",
)

ALLOWED_HMM_SIGNAL_COLUMNS: tuple[str, ...] = (
    "hmm_signal_observation_date",
    "hmm_signal_available_after_close_date",
    "hmm_signal_trade_date",
    "hmm_state_for_next_session",
    "hmm_state_name_for_next_session",
    *ALLOWED_HMM_PROBABILITY_COLUMNS,
)

ALLOWED_MAR_SIGNAL_COLUMNS: tuple[str, ...] = (
    "mar_signal_observation_date",
    "mar_signal_available_after_close_date",
    "mar_signal_trade_date",
    "mar_state_for_next_session",
    "mar_state_name_for_next_session",
    *ALLOWED_MAR_PROBABILITY_COLUMNS,
)


def _normalise_name(value: str) -> str:
    return str(value).strip().lower()


def _normalise_columns(columns: Iterable[str]) -> list[str]:
    return [_normalise_name(col) for col in columns]


def _is_allowed_next_session_column(column: str) -> bool:
    col = _normalise_name(column)
    return any(token in col for token in ALLOWED_NEXT_SESSION_SUBSTRINGS)


def _has_forbidden_suffix(column: str) -> bool:
    col = _normalise_name(column)
    return any(col.endswith(suffix) for suffix in STRATEGY_FORBIDDEN_SUFFIXES)


def is_forbidden_strategy_column(column: str) -> bool:
    """
    Return True if a column is forbidden for Phase 9 strategy construction.

    The policy intentionally allows next-session-safe HMM/MAR columns because
    Phase 6 and Phase 7 already produced next-session-safe signal fields.
    It rejects ex-post labels, smoothed/diagnostic probabilities, crisis labels,
    MSVOL/MSGARCH fields, and forward/future columns.
    """
    col = _normalise_name(column)

    if col in STRATEGY_FORBIDDEN_EXACT_COLUMNS:
        return True

    if _has_forbidden_suffix(col):
        return True

    for token in STRATEGY_FORBIDDEN_FEATURE_SUBSTRINGS:
        if token in col:
            return True

    return False


def get_forbidden_columns(columns: Iterable[str]) -> list[str]:
    """
    Return forbidden columns from an iterable of column names.

    This function only identifies forbidden columns. Later signal-building code
    should drop them immediately, record them in metadata as
    present_but_excluded, and fail only if strategy logic attempts to consume
    them.
    """
    return [col for col in columns if is_forbidden_strategy_column(col)]


def assert_no_strategy_forbidden_columns(columns: Iterable[str]) -> None:
    """
    Raise if any supplied strategy-consumed columns violate no-lookahead policy.

    Use this on the final set of columns consumed by strategy rules, not on raw
    upstream panels. Raw upstream panels may contain forbidden columns that are
    immediately dropped and recorded.
    """
    forbidden = get_forbidden_columns(columns)
    if forbidden:
        raise ValueError(
            "Forbidden Phase 9 strategy input column(s) detected: "
            f"{sorted(forbidden)}"
        )


def assert_strategy_inputs_are_point_in_time(columns: Iterable[str]) -> None:
    """
    Validate that strategy-consumed columns are point-in-time safe.
    """
    assert_no_strategy_forbidden_columns(columns)


def assert_model_allowed_for_strategy(model_name: str) -> None:
    model = _normalise_name(model_name)

    if model in FORBIDDEN_STRATEGY_MODELS:
        raise ValueError(
            f"Model family '{model_name}' is forbidden for active Phase 9 "
            "strategy construction."
        )

    if model not in ALLOWED_STRATEGY_MODELS:
        raise ValueError(
            f"Model family '{model_name}' is not allowed for Phase 9. "
            f"Allowed models: {list(ALLOWED_STRATEGY_MODELS)}."
        )


def assert_no_msvol_strategy_use(
    model_name: str,
    columns: Iterable[str] | None = None,
) -> None:
    model = _normalise_name(model_name)

    if "msvol" in model or "msgarch" in model:
        raise ValueError(
            "MSVOL/MSGARCH cannot be used as active Phase 9 strategy models."
        )

    if columns is not None:
        bad_columns = [
            col
            for col in columns
            if "msvol" in _normalise_name(col) or "msgarch" in _normalise_name(col)
        ]
        if bad_columns:
            raise ValueError(
                "MSVOL/MSGARCH columns cannot be consumed by Phase 9 strategy "
                f"logic: {sorted(bad_columns)}"
            )


def get_allowed_probability_columns(model_name: str) -> tuple[str, ...]:
    model = _normalise_name(model_name)
    assert_model_allowed_for_strategy(model)

    if model == "gaussian_hmm":
        return ALLOWED_HMM_PROBABILITY_COLUMNS

    if model == "markov_autoreg":
        return ALLOWED_MAR_PROBABILITY_COLUMNS

    raise ValueError(
        f"Model family '{model_name}' does not support probabilistic Phase 9 "
        "sizing rules."
    )


def validate_strategy_names(strategy_names: Iterable[str]) -> None:
    supplied = tuple(_normalise_name(name) for name in strategy_names)
    supplied_set = set(supplied)
    approved_set = set(APPROVED_STRATEGY_NAMES)

    rejected_present = supplied_set.intersection(REJECTED_STRATEGY_NAMES)
    if rejected_present:
        raise ValueError(
            "Rejected/deferred Phase 9 strategy name(s) present: "
            f"{sorted(rejected_present)}"
        )

    if supplied_set != approved_set:
        missing = sorted(approved_set.difference(supplied_set))
        extra = sorted(supplied_set.difference(approved_set))
        raise ValueError(
            "Phase 9 strategy universe must contain exactly the approved "
            "seven strategies. "
            f"Missing: {missing}. Extra: {extra}."
        )


def validate_strategy_model_map(strategy_model_map: Mapping[str, str]) -> None:
    validate_strategy_names(strategy_model_map.keys())

    for strategy_name, model_name in strategy_model_map.items():
        strategy = _normalise_name(strategy_name)
        model = _normalise_name(model_name)

        assert_model_allowed_for_strategy(model)
        assert_no_msvol_strategy_use(model)

        if strategy == "unconditional_full" and model != "unconditional":
            raise ValueError("unconditional_full must use model_family='unconditional'.")

        if strategy.startswith("threshold_") and model != "threshold":
            raise ValueError(
                f"{strategy_name} must use model_family='threshold', got '{model_name}'."
            )

        if strategy.startswith("hmm_") and model != "gaussian_hmm":
            raise ValueError(
                f"{strategy_name} must use model_family='gaussian_hmm', "
                f"got '{model_name}'."
            )

        if strategy.startswith("mar_") and model != "markov_autoreg":
            raise ValueError(
                f"{strategy_name} must use model_family='markov_autoreg', "
                f"got '{model_name}'."
            )