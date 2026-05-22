from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from vrp.strategies.exposure_rules import (
    BLOCK_NONE,
    DECISION_UNAVAILABLE,
    ExposureDecision,
    available_decision,
    threshold_defensive_exposure,
    threshold_hard_filter_exposure,
    unconditional_full_exposure,
    unavailable_decision,
    probability_linear_decision,
    apply_probability_carry_gate,
)
from vrp.strategies.signal_schema import (
    DATE_COLUMN,
    PHASE9_OUTPUT_COLUMNS,
    REQUIRED_HAR_COLUMNS,
    REQUIRED_HMM_COLUMNS,
    REQUIRED_MAR_COLUMNS,
    REQUIRED_THRESHOLD_COLUMNS,
    normalise_market,
    require_source_columns,
    sanitize_input_frames,
    validate_phase9_signal_panel,
)
from vrp.strategies.strategy_config import StrategyDefinition
from vrp.strategies.strategy_registry import APPROVED_STRATEGY_NAMES


MISSING_TARGET_TRADE_DATE = "missing_target_trade_date"
MISSING_THRESHOLD_REGIME = "missing_threshold_regime"
INVALID_THRESHOLD_STATE = "invalid_threshold_state"


@dataclass(frozen=True)
class SignalBuildResult:
    signals: pd.DataFrame
    forbidden_columns_present_but_excluded: dict[str, tuple[str, ...]]
    forbidden_columns_used: tuple[str, ...]


def default_strategy_definitions() -> dict[str, StrategyDefinition]:
    """
    Default approved Phase 9 strategy definitions.

    CLI code should normally pass definitions loaded from configs/strategies.yaml.
    This fallback keeps tests and interactive use simple while still enforcing
    the exact approved seven-strategy universe.
    """
    return {
        "unconditional_full": StrategyDefinition(
            name="unconditional_full",
            active=True,
            model_family="unconditional",
            rule_name="unconditional_full",
            requires_probabilities=False,
            requires_har=False,
            probability_rule=None,
            transition_exposure=None,
            max_stress_probability=None,
            require_vrp_har_positive=False,
            description="Unconditional full short-vol benchmark.",
        ),
        "threshold_hard_filter": StrategyDefinition(
            name="threshold_hard_filter",
            active=True,
            model_family="threshold",
            rule_name="hard_filter",
            requires_probabilities=False,
            requires_har=False,
            probability_rule=None,
            transition_exposure=None,
            max_stress_probability=None,
            require_vrp_har_positive=False,
            description="Deterministic threshold stress-veto baseline.",
        ),
        "threshold_defensive": StrategyDefinition(
            name="threshold_defensive",
            active=True,
            model_family="threshold",
            rule_name="defensive",
            requires_probabilities=False,
            requires_har=False,
            probability_rule=None,
            transition_exposure=-0.25,
            max_stress_probability=None,
            require_vrp_har_positive=False,
            description="Threshold baseline with partial transition exposure.",
        ),
        "hmm_prob_linear": StrategyDefinition(
            name="hmm_prob_linear",
            active=True,
            model_family="gaussian_hmm",
            rule_name="probability_linear_margin",
            requires_probabilities=True,
            requires_har=False,
            probability_rule="calm_minus_stress",
            transition_exposure=None,
            max_stress_probability=None,
            require_vrp_har_positive=False,
            description="HMM filtered-probability linear exposure sizing.",
        ),
        "hmm_prob_linear_carry": StrategyDefinition(
            name="hmm_prob_linear_carry",
            active=True,
            model_family="gaussian_hmm",
            rule_name="probability_linear_margin_carry",
            requires_probabilities=True,
            requires_har=True,
            probability_rule="calm_minus_stress",
            transition_exposure=None,
            max_stress_probability=0.40,
            require_vrp_har_positive=True,
            description="HMM probability sizing with positive prospective VRP gate.",
        ),
        "mar_prob_linear": StrategyDefinition(
            name="mar_prob_linear",
            active=True,
            model_family="markov_autoreg",
            rule_name="probability_linear_margin",
            requires_probabilities=True,
            requires_har=False,
            probability_rule="calm_minus_stress",
            transition_exposure=None,
            max_stress_probability=None,
            require_vrp_har_positive=False,
            description="MAR filtered-probability linear exposure sizing.",
        ),
        "mar_prob_linear_carry": StrategyDefinition(
            name="mar_prob_linear_carry",
            active=True,
            model_family="markov_autoreg",
            rule_name="probability_linear_margin_carry",
            requires_probabilities=True,
            requires_har=True,
            probability_rule="calm_minus_stress",
            transition_exposure=None,
            max_stress_probability=0.40,
            require_vrp_har_positive=True,
            description="MAR probability sizing with positive prospective VRP gate.",
        ),
    }


def _filter_strategy_definitions(
    strategy_definitions: Mapping[str, StrategyDefinition],
    requested_strategy: str = "all",
) -> dict[str, StrategyDefinition]:
    active = {
        name: definition
        for name, definition in strategy_definitions.items()
        if definition.active
    }

    approved = set(APPROVED_STRATEGY_NAMES)
    active_names = set(active)

    rejected_or_extra = sorted(active_names.difference(approved))
    if rejected_or_extra:
        raise ValueError(
            "Phase 9 received rejected or unapproved strategy definition(s): "
            f"{rejected_or_extra}"
        )

    if requested_strategy == "all":
        missing = sorted(approved.difference(active_names))
        if missing:
            raise ValueError(
                "Phase 9 strategy universe must contain all seven approved "
                f"strategies. Missing: {missing}."
            )
        return {name: active[name] for name in APPROVED_STRATEGY_NAMES}

    if requested_strategy not in approved:
        raise ValueError(
            f"Requested strategy '{requested_strategy}' is not approved. "
            f"Approved strategies: {sorted(approved)}."
        )

    if requested_strategy not in active:
        raise ValueError(f"Requested strategy '{requested_strategy}' is not active.")

    return {requested_strategy: active[requested_strategy]}


def _copy_with_datetime(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")

    return result


def _prepare_inputs(
    har: pd.DataFrame,
    threshold: pd.DataFrame,
    hmm: pd.DataFrame,
    mar: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[str, ...]]]:
    frames = {
        "har": har,
        "threshold": threshold,
        "gaussian_hmm": hmm,
        "markov_autoreg": mar,
    }

    sanitized_frames, forbidden_present = sanitize_input_frames(frames)

    require_source_columns(sanitized_frames["har"], "har", frame_name="har")
    require_source_columns(
        sanitized_frames["threshold"],
        "threshold",
        frame_name="threshold",
    )
    require_source_columns(
        sanitized_frames["gaussian_hmm"],
        "gaussian_hmm",
        frame_name="gaussian_hmm",
    )
    require_source_columns(
        sanitized_frames["markov_autoreg"],
        "markov_autoreg",
        frame_name="markov_autoreg",
    )

    sanitized_frames["har"] = _copy_with_datetime(
        sanitized_frames["har"],
        [DATE_COLUMN],
    )
    sanitized_frames["threshold"] = _copy_with_datetime(
        sanitized_frames["threshold"],
        [DATE_COLUMN],
    )
    sanitized_frames["gaussian_hmm"] = _copy_with_datetime(
        sanitized_frames["gaussian_hmm"],
        [
            "hmm_signal_observation_date",
            "hmm_signal_available_after_close_date",
            "hmm_signal_trade_date",
        ],
    )
    sanitized_frames["markov_autoreg"] = _copy_with_datetime(
        sanitized_frames["markov_autoreg"],
        [
            "mar_signal_observation_date",
            "mar_signal_available_after_close_date",
            "mar_signal_trade_date",
        ],
    )

    return sanitized_frames, forbidden_present


def _sorted_unique_dates(series: pd.Series) -> list[pd.Timestamp]:
    dates = pd.to_datetime(series, errors="coerce").dropna().drop_duplicates()
    return sorted(pd.Timestamp(value) for value in dates)


def _next_trade_date_map(dates: Iterable[pd.Timestamp]) -> dict[pd.Timestamp, Optional[pd.Timestamp]]:
    sorted_dates = sorted(pd.Timestamp(value) for value in dates if pd.notna(value))

    mapping: dict[pd.Timestamp, Optional[pd.Timestamp]] = {}
    for index, current_date in enumerate(sorted_dates):
        if index + 1 < len(sorted_dates):
            mapping[current_date] = sorted_dates[index + 1]
        else:
            mapping[current_date] = None

    return mapping


def _har_lookup(har: pd.DataFrame) -> dict[pd.Timestamp, dict[str, Any]]:
    lookup: dict[pd.Timestamp, dict[str, Any]] = {}

    for _, row in har.iterrows():
        date_value = row.get(DATE_COLUMN)
        if pd.isna(date_value):
            continue
        row_dict: dict[str, Any] = {str(key): value for key, value in row.to_dict().items()}
        lookup[pd.Timestamp(date_value)] = row_dict

    return lookup


def _har_values_for_date(
    lookup: Mapping[pd.Timestamp, Mapping[str, Any]],
    observation_date: Any,
) -> tuple[Any, Any]:
    if pd.isna(observation_date):
        return pd.NA, pd.NA

    values = lookup.get(pd.Timestamp(observation_date))
    if values is None:
        return pd.NA, pd.NA

    return (
        values.get("vrp_har_gk", pd.NA),
        values.get("har_forecast_available", pd.NA),
    )


def _strict_python_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _record(
    *,
    market: str,
    strategy_name: str,
    regime_model: str,
    signal_observation_date: Any,
    signal_available_after_close_date: Any,
    target_trade_date: Any,
    decision: ExposureDecision,
    state_name: Any = pd.NA,
    p_calm: Any = pd.NA,
    p_transition: Any = pd.NA,
    p_stress: Any = pd.NA,
    vrp_har_gk: Any = pd.NA,
    har_forecast_available: Any = pd.NA,
    source_signal_date_column: str,
    source_model: str,
) -> dict[str, Any]:
    return {
        "market": market,
        "strategy_name": strategy_name,
        "regime_model": regime_model,
        "signal_observation_date": signal_observation_date,
        "signal_available_after_close_date": signal_available_after_close_date,
        "target_trade_date": target_trade_date,
        "target_exposure": decision.target_exposure,
        "strategy_available": decision.strategy_available,
        "blocked_reason": decision.blocked_reason,
        "decision_reason": decision.decision_reason,
        "state_name": state_name,
        "p_calm": p_calm,
        "p_transition": p_transition,
        "p_stress": p_stress,
        "vrp_har_gk": vrp_har_gk,
        "har_forecast_available": har_forecast_available,
        "source_signal_date_column": source_signal_date_column,
        "source_model": source_model,
    }


def _decision_for_missing_trade_date(target_trade_date: Any) -> ExposureDecision | None:
    if pd.isna(target_trade_date):
        return unavailable_decision(MISSING_TARGET_TRADE_DATE)
    return None


def _build_unconditional_records(
    *,
    market: str,
    har: pd.DataFrame,
    strategy: StrategyDefinition,
) -> list[dict[str, Any]]:
    dates = _sorted_unique_dates(har[DATE_COLUMN])
    next_map = _next_trade_date_map(dates)
    har_by_date = _har_lookup(har)

    records: list[dict[str, Any]] = []

    for observation_date in dates:
        target_trade_date = next_map[observation_date]
        vrp_har_gk, har_available = _har_values_for_date(
            har_by_date,
            observation_date,
        )

        missing_trade_date_decision = _decision_for_missing_trade_date(
            target_trade_date,
        )

        if missing_trade_date_decision is not None:
            decision = missing_trade_date_decision
        else:
            decision = available_decision(
                unconditional_full_exposure(),
                "unconditional_full",
            )

        records.append(
            _record(
                market=market,
                strategy_name=strategy.name,
                regime_model=strategy.model_family,
                signal_observation_date=observation_date,
                signal_available_after_close_date=observation_date,
                target_trade_date=target_trade_date,
                decision=decision,
                vrp_har_gk=vrp_har_gk,
                har_forecast_available=har_available,
                source_signal_date_column=DATE_COLUMN,
                source_model="unconditional",
            )
        )

    return records


def _threshold_decision(
    *,
    strategy_name: str,
    state_name: Any,
    threshold_regime_available: Any,
    transition_exposure: float | None,
) -> ExposureDecision:
    if not _strict_python_true(threshold_regime_available):
        return unavailable_decision(MISSING_THRESHOLD_REGIME)

    try:
        if strategy_name == "threshold_hard_filter":
            exposure = threshold_hard_filter_exposure(state_name)
            reason = (
                "stress_veto"
                if str(state_name).strip().lower() == "stress"
                else "non_stress_full_short_vol"
            )
        elif strategy_name == "threshold_defensive":
            exposure = threshold_defensive_exposure(
                state_name,
                transition_exposure=-0.25 if transition_exposure is None else transition_exposure,
            )

            state = str(state_name).strip().lower()
            if state == "calm":
                reason = "calm_full_short_vol"
            elif state == "transition":
                reason = "transition_partial_exposure"
            else:
                reason = "stress_defensive_flat"
        else:
            raise ValueError(f"Unsupported threshold strategy: {strategy_name}")
    except ValueError:
        return unavailable_decision(INVALID_THRESHOLD_STATE)

    return available_decision(exposure, reason)


def _build_threshold_records(
    *,
    market: str,
    threshold: pd.DataFrame,
    har: pd.DataFrame,
    strategies: Mapping[str, StrategyDefinition],
) -> list[dict[str, Any]]:
    dates = _sorted_unique_dates(threshold[DATE_COLUMN])
    next_map = _next_trade_date_map(dates)
    har_by_date = _har_lookup(har)

    threshold_sorted = threshold.sort_values(DATE_COLUMN).copy()
    records: list[dict[str, Any]] = []

    for _, row in threshold_sorted.iterrows():
        observation_date = row.get(DATE_COLUMN)
        if pd.isna(observation_date):
            continue

        observation_date = pd.Timestamp(observation_date)
        target_trade_date = next_map.get(observation_date, pd.NaT)
        vrp_har_gk, har_available = _har_values_for_date(
            har_by_date,
            observation_date,
        )

        for strategy_name in ("threshold_hard_filter", "threshold_defensive"):
            strategy = strategies.get(strategy_name)
            if strategy is None:
                continue

            missing_trade_date_decision = _decision_for_missing_trade_date(
                target_trade_date,
            )

            if missing_trade_date_decision is not None:
                decision = missing_trade_date_decision
            else:
                decision = _threshold_decision(
                    strategy_name=strategy_name,
                    state_name=row.get("threshold_state_name"),
                    threshold_regime_available=row.get("threshold_regime_available"),
                    transition_exposure=strategy.transition_exposure,
                )

            records.append(
                _record(
                    market=market,
                    strategy_name=strategy.name,
                    regime_model=strategy.model_family,
                    signal_observation_date=observation_date,
                    signal_available_after_close_date=observation_date,
                    target_trade_date=target_trade_date,
                    decision=decision,
                    state_name=row.get("threshold_state_name", pd.NA),
                    vrp_har_gk=vrp_har_gk,
                    har_forecast_available=har_available,
                    source_signal_date_column=DATE_COLUMN,
                    source_model="threshold",
                )
            )

    return records


def _probability_decision(
    *,
    strategy: StrategyDefinition,
    p_calm: Any,
    p_transition: Any,
    p_stress: Any,
    vrp_har_gk: Any,
    har_forecast_available: Any,
) -> ExposureDecision:
    if strategy.requires_har:
        return apply_probability_carry_gate(
            p_calm=p_calm,
            p_transition=p_transition,
            p_stress=p_stress,
            vrp_har_gk=vrp_har_gk,
            har_forecast_available=_strict_python_true(har_forecast_available),
            stress_probability_cutoff=(
                0.40
                if strategy.max_stress_probability is None
                else strategy.max_stress_probability
            ),
        )

    return probability_linear_decision(
        p_calm=p_calm,
        p_transition=p_transition,
        p_stress=p_stress,
    )


def _build_hmm_records(
    *,
    market: str,
    hmm: pd.DataFrame,
    har: pd.DataFrame,
    strategies: Mapping[str, StrategyDefinition],
) -> list[dict[str, Any]]:
    har_by_date = _har_lookup(har)
    hmm_sorted = hmm.sort_values("hmm_signal_observation_date").copy()

    records: list[dict[str, Any]] = []

    for _, row in hmm_sorted.iterrows():
        observation_date = row.get("hmm_signal_observation_date")
        after_close_date = row.get("hmm_signal_available_after_close_date")
        target_trade_date = row.get("hmm_signal_trade_date")

        if pd.notna(observation_date):
            observation_date = pd.Timestamp(observation_date)

        vrp_har_gk, har_available = _har_values_for_date(
            har_by_date,
            observation_date,
        )

        p_calm = row.get("hmm_filtered_prob_calm_for_next_session")
        p_transition = row.get("hmm_filtered_prob_transition_for_next_session")
        p_stress = row.get("hmm_filtered_prob_stress_for_next_session")

        for strategy_name in ("hmm_prob_linear", "hmm_prob_linear_carry"):
            strategy = strategies.get(strategy_name)
            if strategy is None:
                continue

            missing_trade_date_decision = _decision_for_missing_trade_date(
                target_trade_date,
            )

            if missing_trade_date_decision is not None:
                decision = missing_trade_date_decision
            else:
                decision = _probability_decision(
                    strategy=strategy,
                    p_calm=p_calm,
                    p_transition=p_transition,
                    p_stress=p_stress,
                    vrp_har_gk=vrp_har_gk,
                    har_forecast_available=har_available,
                )

            records.append(
                _record(
                    market=market,
                    strategy_name=strategy.name,
                    regime_model=strategy.model_family,
                    signal_observation_date=observation_date,
                    signal_available_after_close_date=after_close_date,
                    target_trade_date=target_trade_date,
                    decision=decision,
                    state_name=row.get("hmm_state_name_for_next_session", pd.NA),
                    p_calm=p_calm,
                    p_transition=p_transition,
                    p_stress=p_stress,
                    vrp_har_gk=vrp_har_gk,
                    har_forecast_available=har_available,
                    source_signal_date_column="hmm_signal_observation_date",
                    source_model="gaussian_hmm",
                )
            )

    return records


def _build_mar_records(
    *,
    market: str,
    mar: pd.DataFrame,
    har: pd.DataFrame,
    strategies: Mapping[str, StrategyDefinition],
) -> list[dict[str, Any]]:
    har_by_date = _har_lookup(har)
    mar_sorted = mar.sort_values("mar_signal_observation_date").copy()

    records: list[dict[str, Any]] = []

    for _, row in mar_sorted.iterrows():
        observation_date = row.get("mar_signal_observation_date")
        after_close_date = row.get("mar_signal_available_after_close_date")
        target_trade_date = row.get("mar_signal_trade_date")

        if pd.notna(observation_date):
            observation_date = pd.Timestamp(observation_date)

        vrp_har_gk, har_available = _har_values_for_date(
            har_by_date,
            observation_date,
        )

        p_calm = row.get("mar_filtered_prob_calm_for_next_session")
        p_transition = row.get("mar_filtered_prob_transition_for_next_session")
        p_stress = row.get("mar_filtered_prob_stress_for_next_session")

        for strategy_name in ("mar_prob_linear", "mar_prob_linear_carry"):
            strategy = strategies.get(strategy_name)
            if strategy is None:
                continue

            missing_trade_date_decision = _decision_for_missing_trade_date(
                target_trade_date,
            )

            if missing_trade_date_decision is not None:
                decision = missing_trade_date_decision
            else:
                decision = _probability_decision(
                    strategy=strategy,
                    p_calm=p_calm,
                    p_transition=p_transition,
                    p_stress=p_stress,
                    vrp_har_gk=vrp_har_gk,
                    har_forecast_available=har_available,
                )

            records.append(
                _record(
                    market=market,
                    strategy_name=strategy.name,
                    regime_model=strategy.model_family,
                    signal_observation_date=observation_date,
                    signal_available_after_close_date=after_close_date,
                    target_trade_date=target_trade_date,
                    decision=decision,
                    state_name=row.get("mar_state_name_for_next_session", pd.NA),
                    p_calm=p_calm,
                    p_transition=p_transition,
                    p_stress=p_stress,
                    vrp_har_gk=vrp_har_gk,
                    har_forecast_available=har_available,
                    source_signal_date_column="mar_signal_observation_date",
                    source_model="markov_autoreg",
                )
            )

    return records


def build_phase9_signal_panel(
    *,
    market: str,
    har: pd.DataFrame,
    threshold: pd.DataFrame,
    hmm: pd.DataFrame,
    mar: pd.DataFrame,
    strategy_definitions: Mapping[str, StrategyDefinition] | None = None,
    requested_strategy: str = "all",
    validate_output: bool = True,
) -> SignalBuildResult:
    """
    Build the Phase 9 long-format strategy signal panel.

    This function does not read files and does not write files. It only converts
    already-produced Phase 4/5/6/7 panels into next-session exposure intentions.
    """
    market_key = normalise_market(market)

    definitions = (
        default_strategy_definitions()
        if strategy_definitions is None
        else dict(strategy_definitions)
    )
    active_strategies = _filter_strategy_definitions(
        definitions,
        requested_strategy=requested_strategy,
    )

    sanitized, forbidden_present = _prepare_inputs(
        har=har,
        threshold=threshold,
        hmm=hmm,
        mar=mar,
    )

    records: list[dict[str, Any]] = []

    if "unconditional_full" in active_strategies:
        records.extend(
            _build_unconditional_records(
                market=market_key,
                har=sanitized["har"],
                strategy=active_strategies["unconditional_full"],
            )
        )

    if {
        "threshold_hard_filter",
        "threshold_defensive",
    }.intersection(active_strategies):
        records.extend(
            _build_threshold_records(
                market=market_key,
                threshold=sanitized["threshold"],
                har=sanitized["har"],
                strategies=active_strategies,
            )
        )

    if {"hmm_prob_linear", "hmm_prob_linear_carry"}.intersection(active_strategies):
        records.extend(
            _build_hmm_records(
                market=market_key,
                hmm=sanitized["gaussian_hmm"],
                har=sanitized["har"],
                strategies=active_strategies,
            )
        )

    if {"mar_prob_linear", "mar_prob_linear_carry"}.intersection(active_strategies):
        records.extend(
            _build_mar_records(
                market=market_key,
                mar=sanitized["markov_autoreg"],
                har=sanitized["har"],
                strategies=active_strategies,
            )
        )

    signal_panel = pd.DataFrame.from_records(records, columns=list(PHASE9_OUTPUT_COLUMNS))

    for column in (
        "signal_observation_date",
        "signal_available_after_close_date",
        "target_trade_date",
    ):
        signal_panel[column] = pd.to_datetime(signal_panel[column], errors="coerce")

    if validate_output:
        validate_phase9_signal_panel(signal_panel)

    return SignalBuildResult(
        signals=signal_panel,
        forbidden_columns_present_but_excluded=forbidden_present,
        forbidden_columns_used=(),
    )


__all__ = [
    "MISSING_TARGET_TRADE_DATE",
    "MISSING_THRESHOLD_REGIME",
    "INVALID_THRESHOLD_STATE",
    "SignalBuildResult",
    "default_strategy_definitions",
    "build_phase9_signal_panel",
]