from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, cast

import pandas as pd


CALM: int = 0
TRANSITION: int = 1
STRESS: int = 2

STATE_ID_TO_NAME: Dict[int, str] = {
    CALM: "calm",
    TRANSITION: "transition",
    STRESS: "stress",
}

STATE_NAME_TO_ID: Dict[str, int] = {
    "calm": CALM,
    "transition": TRANSITION,
    "stress": STRESS,
}

VALID_STATE_IDS = frozenset(STATE_ID_TO_NAME.keys())
VALID_STATE_NAMES = frozenset(STATE_NAME_TO_ID.keys())


def _normalise_state_name(value: object) -> object:
    if _is_missing(value):
        return pd.NA
    return str(value).strip().lower()


def _is_missing(value: object) -> bool:
    return bool(pd.isna(cast(Any, value)))


def validate_state_series(series: pd.Series, allow_missing: bool = True) -> bool:
    """
    Validate that a pandas Series contains only the canonical regime IDs:
    calm=0, transition=1, stress=2.

    Missing values are allowed by default because component states may be unavailable
    when history, price data, or HAR forecasts are missing.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("validate_state_series expects a pandas Series.")

    values = series.copy()

    if allow_missing:
        values = values.dropna()
    elif values.isna().any():
        bad_count = int(values.isna().sum())
        raise ValueError(f"State series contains {bad_count} missing value(s).")

    if values.empty:
        return True

    invalid_values = sorted(
        {
            value
            for value in values.tolist()
            if value not in VALID_STATE_IDS
        }
    )

    if invalid_values:
        raise ValueError(
            "State series contains invalid state ID(s): "
            f"{invalid_values}. Valid IDs are {sorted(VALID_STATE_IDS)}."
        )

    return True


def validate_state_name_series(series: pd.Series, allow_missing: bool = True) -> bool:
    """
    Validate that a pandas Series contains only canonical regime names:
    calm, transition, stress.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("validate_state_name_series expects a pandas Series.")

    values = series.map(_normalise_state_name)

    if allow_missing:
        values = values.dropna()
    elif values.isna().any():
        bad_count = int(values.isna().sum())
        raise ValueError(f"State-name series contains {bad_count} missing value(s).")

    if values.empty:
        return True

    invalid_values = sorted(
        {
            value
            for value in values.tolist()
            if value not in VALID_STATE_NAMES
        }
    )

    if invalid_values:
        raise ValueError(
            "State-name series contains invalid state name(s): "
            f"{invalid_values}. Valid names are {sorted(VALID_STATE_NAMES)}."
        )

    return True


def map_state_id_to_name(series: pd.Series) -> pd.Series:
    """
    Map numeric state IDs to state names.

    Missing values are preserved.
    """
    validate_state_series(series, allow_missing=True)

    return series.map(
        lambda value: pd.NA if _is_missing(value) else STATE_ID_TO_NAME[cast(int, value)]
    ).astype("object")


def map_state_name_to_id(series: pd.Series) -> pd.Series:
    """
    Map state names to numeric state IDs.

    Missing values are preserved. Output uses pandas nullable Int64 dtype.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("map_state_name_to_id expects a pandas Series.")

    normalised = series.map(_normalise_state_name)
    validate_state_name_series(normalised, allow_missing=True)

    mapped = normalised.map(
        lambda value: pd.NA if _is_missing(value) else STATE_NAME_TO_ID[str(value)]
    )

    return mapped.astype("Int64")


def state_id_to_name(state_id: int) -> str:
    """
    Map a single state ID to its canonical state name.
    """
    if state_id not in STATE_ID_TO_NAME:
        raise ValueError(
            f"Invalid state ID {state_id}. Valid IDs are {sorted(VALID_STATE_IDS)}."
        )
    return STATE_ID_TO_NAME[state_id]


def state_name_to_id(state_name: str) -> int:
    """
    Map a single state name to its canonical state ID.
    """
    normalised = str(state_name).strip().lower()
    if normalised not in STATE_NAME_TO_ID:
        raise ValueError(
            f"Invalid state name {state_name!r}. "
            f"Valid names are {sorted(VALID_STATE_NAMES)}."
        )
    return STATE_NAME_TO_ID[normalised]


def validate_state_mapping_consistency(
    id_to_name: Mapping[int, str] = STATE_ID_TO_NAME,
    name_to_id: Mapping[str, int] = STATE_NAME_TO_ID,
) -> bool:
    """
    Validate that ID/name mappings are reversible.
    """
    for state_id, state_name in id_to_name.items():
        if name_to_id.get(state_name) != state_id:
            raise ValueError(
                f"State mapping is not reversible for {state_id} -> {state_name}."
            )

    for state_name, state_id in name_to_id.items():
        if id_to_name.get(state_id) != state_name:
            raise ValueError(
                f"State mapping is not reversible for {state_name} -> {state_id}."
            )

    return True