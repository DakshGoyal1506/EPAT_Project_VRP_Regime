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

# ---------------------------------------------------------------------------
# Phase 6 Gaussian HMM economic state labeling
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vrp.regimes.hmm_registry import (
    HMM_ECONOMIC_STATE_NAMES,
    HMM_FILTERED_ECONOMIC_PROB_COLUMNS,
    HMM_SIGNAL_AVAILABILITY_COLUMNS,
    HMM_TRANSITION_STATE_MODELLED_COLUMN,
    get_hmm_filtered_economic_probability_columns,
)


HMM_RAW_STATE_COL = "hmm_raw_state"
HMM_STATE_COL = "hmm_state"
HMM_STATE_NAME_COL = "hmm_state_name"
HMM_STATE_STRESS_SCORE_COL = "hmm_state_stress_score"
HMM_STATE_MAPPING_VALID_COL = "hmm_state_mapping_valid"
HMM_STATE_MAPPING_REASON_COL = "hmm_state_mapping_reason"

HMM_ECONOMIC_STATE_TO_ID = {
    "calm": 0,
    "transition": 1,
    "stress": 2,
}

HMM_ECONOMIC_ID_TO_STATE = {
    value: key for key, value in HMM_ECONOMIC_STATE_TO_ID.items()
}

HMM_DEFAULT_STRESS_SCORE_COMPONENTS = {
    "mean_iv_ann": {
        "feature": "iv_ann",
        "direction": "high",
    },
    "mean_rv_gk_22d_ann_lag1": {
        "feature": "rv_gk_22d_ann_lag1",
        "direction": "high",
    },
    "mean_index_return": {
        "feature": "index_return",
        "direction": "low",
    },
    "mean_vrp_har_gk": {
        "feature": "vrp_har_gk",
        "direction": "low",
    },
}


@dataclass(frozen=True)
class HMMStateLabelingResult:
    """
    Economic mapping result for one fitted HMM candidate.

    raw_state_to_name:
        Mapping from raw HMM state id to economic state name:
        calm / transition / stress.

    raw_state_to_id:
        Mapping from raw HMM state id to economic state id:
        calm=0, transition=1, stress=2.

    transition_state_modelled:
        True for K=3, False for K=2.

    economic_monotonicity_passed:
        True only when stress-score ordering is unique and interpretable.

    state_properties:
        Per-raw-state train-period properties and stress scores.
    """

    n_states: int
    raw_state_to_name: Mapping[int, str]
    raw_state_to_id: Mapping[int, int]
    transition_state_modelled: bool
    economic_monotonicity_passed: bool
    rejection_reason: str
    state_properties: pd.DataFrame

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serialisable state-label metadata."""
        return {
            "n_states": int(self.n_states),
            "raw_state_to_name": {
                str(k): str(v) for k, v in self.raw_state_to_name.items()
            },
            "raw_state_to_id": {
                str(k): int(v) for k, v in self.raw_state_to_id.items()
            },
            "transition_state_modelled": bool(self.transition_state_modelled),
            "economic_monotonicity_passed": bool(self.economic_monotonicity_passed),
            "rejection_reason": str(self.rejection_reason),
        }


def _validate_n_states(n_states: int) -> None:
    """Validate Phase 6 state count."""
    if n_states not in {2, 3}:
        raise ValueError(f"Phase 6 supports only 2-state or 3-state HMMs. Got {n_states}.")


def _validate_raw_states(raw_states: Sequence[int] | np.ndarray, *, n_states: int) -> np.ndarray:
    """Validate raw HMM state labels."""
    _validate_n_states(n_states)

    arr = np.asarray(raw_states)

    if arr.ndim != 1:
        raise ValueError(f"raw_states must be 1D. Got shape {arr.shape}.")

    if arr.size == 0:
        raise ValueError("raw_states cannot be empty.")

    if not np.issubdtype(arr.dtype, np.integer):
        if np.all(np.equal(arr, arr.astype(int))):
            arr = arr.astype(int)
        else:
            raise ValueError("raw_states must contain integer state labels.")

    invalid = sorted(set(arr.tolist()) - set(range(n_states)))
    if invalid:
        raise ValueError(f"raw_states contain invalid labels for n_states={n_states}: {invalid}")

    return arr.astype(int)


def _validate_train_indices(
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_rows: int,
) -> np.ndarray:
    """Validate train-window positional indices."""
    idx = np.asarray(train_indices, dtype=int)

    if idx.ndim != 1:
        raise ValueError(f"train_indices must be 1D. Got shape {idx.shape}.")

    if idx.size == 0:
        raise ValueError("train_indices cannot be empty.")

    if idx.min() < 0 or idx.max() >= n_rows:
        raise ValueError("train_indices are out of bounds.")

    if np.any(np.diff(idx) < 0):
        raise ValueError("train_indices must be sorted in chronological order.")

    return idx


def _coerce_numeric_feature(
    df: pd.DataFrame,
    feature: str,
) -> pd.Series:
    """Return numeric feature series."""
    if feature not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)

    return pd.to_numeric(df[feature], errors="coerce").astype(float)


def compute_train_state_properties(
    panel: pd.DataFrame,
    raw_states: Sequence[int] | np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_states: int,
    stress_score_components: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute train-period raw-state economic properties.

    This function uses train rows only. It must not receive full-sample states for
    state mapping logic.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    _validate_n_states(n_states)
    states = _validate_raw_states(raw_states, n_states=n_states)

    if len(panel) != len(states):
        raise ValueError(
            f"panel/raw_states length mismatch: {len(panel)} vs {len(states)}."
        )

    idx = _validate_train_indices(train_indices, n_rows=len(panel))

    components = stress_score_components or HMM_DEFAULT_STRESS_SCORE_COMPONENTS

    train_panel = panel.iloc[idx].copy()
    train_states = states[idx]

    rows: list[dict[str, Any]] = []
    for state in range(n_states):
        state_mask = train_states == state
        state_rows = train_panel.loc[state_mask]

        row: dict[str, Any] = {
            "raw_state": state,
            "n_train_observations": int(state_mask.sum()),
            "train_occupancy": (
                float(state_mask.mean()) if len(train_states) > 0 else np.nan
            ),
        }

        for property_name, spec in components.items():
            feature = spec["feature"]
            values = _coerce_numeric_feature(state_rows, feature)
            row[property_name] = float(values.mean()) if values.notna().any() else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_state_stress_scores(
    state_properties: pd.DataFrame,
    *,
    stress_score_components: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Add rank-based stress score to raw-state property table.

    Direction semantics:
    - high: larger property value gets larger stress rank
    - low: smaller property value gets larger stress rank
    """
    if not isinstance(state_properties, pd.DataFrame):
        raise TypeError("state_properties must be a pandas DataFrame.")

    if "raw_state" not in state_properties.columns:
        raise ValueError("state_properties must contain raw_state column.")

    components = stress_score_components or HMM_DEFAULT_STRESS_SCORE_COMPONENTS
    out = state_properties.copy()

    rank_cols: list[str] = []
    missing_components: list[str] = []

    for property_name, spec in components.items():
        direction = spec["direction"].strip().lower()

        if property_name not in out.columns:
            missing_components.append(property_name)
            continue

        values = pd.to_numeric(out[property_name], errors="coerce")

        if values.isna().any():
            missing_components.append(property_name)
            continue

        if direction == "high":
            ranks = values.rank(method="average", ascending=True)
        elif direction == "low":
            ranks = values.rank(method="average", ascending=False)
        else:
            raise ValueError(
                f"Unsupported stress-score direction for {property_name!r}: {direction!r}."
            )

        rank_col = f"{property_name}_stress_rank"
        out[rank_col] = ranks.astype(float)
        rank_cols.append(rank_col)

    if missing_components:
        out["hmm_state_stress_score"] = np.nan
        out["hmm_state_score_valid"] = False
        out["hmm_state_score_reason"] = (
            "missing_or_nan_components:" + ",".join(missing_components)
        )
        return out

    out["hmm_state_stress_score"] = out[rank_cols].sum(axis=1).astype(float)
    out["hmm_state_score_valid"] = True
    out["hmm_state_score_reason"] = ""

    return out


def _mapping_from_scores(
    scored: pd.DataFrame,
    *,
    n_states: int,
) -> tuple[dict[int, str], dict[int, int], bool, str]:
    """
    Build raw-state -> economic-state mapping from stress scores.

    Ties are mapped deterministically by raw_state order but marked invalid.
    """
    _validate_n_states(n_states)

    required = {"raw_state", "hmm_state_stress_score", "hmm_state_score_valid"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"scored state table missing columns: {missing}")

    if not bool(scored["hmm_state_score_valid"].all()):
        reason = "invalid_stress_score_components"
        ordered = scored.sort_values(["raw_state"]).copy()
        raw_states = ordered["raw_state"].astype(int).tolist()

        if n_states == 2:
            raw_to_name = {
                raw_states[0]: "calm",
                raw_states[1]: "stress",
            }
        else:
            raw_to_name = {
                raw_states[0]: "calm",
                raw_states[1]: "transition",
                raw_states[2]: "stress",
            }

        raw_to_id = {
            raw: HMM_ECONOMIC_STATE_TO_ID[name]
            for raw, name in raw_to_name.items()
        }
        return raw_to_name, raw_to_id, False, reason

    ordered = scored.sort_values(
        ["hmm_state_stress_score", "raw_state"],
        ascending=[True, True],
    ).copy()

    scores = ordered["hmm_state_stress_score"].astype(float).to_numpy()
    raw_states = ordered["raw_state"].astype(int).tolist()

    unique_scores = len(np.unique(scores)) == len(scores)
    monotonicity_passed = bool(unique_scores)

    reason = "" if monotonicity_passed else "tied_stress_scores"

    if n_states == 2:
        raw_to_name = {
            raw_states[0]: "calm",
            raw_states[1]: "stress",
        }
    else:
        raw_to_name = {
            raw_states[0]: "calm",
            raw_states[1]: "transition",
            raw_states[2]: "stress",
        }

    raw_to_id = {
        raw_state: HMM_ECONOMIC_STATE_TO_ID[name]
        for raw_state, name in raw_to_name.items()
    }

    return raw_to_name, raw_to_id, monotonicity_passed, reason


def label_hmm_states_from_train_properties(
    panel: pd.DataFrame,
    raw_states: Sequence[int] | np.ndarray,
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_states: int,
    stress_score_components: Mapping[str, Mapping[str, str]] | None = None,
) -> HMMStateLabelingResult:
    """
    Label raw HMM states as calm / transition / stress using train rows only.

    For K=3:
    - lowest stress_score -> calm
    - middle stress_score -> transition
    - highest stress_score -> stress

    For K=2:
    - lowest stress_score -> calm
    - highest stress_score -> stress
    - transition is not modelled
    """
    _validate_n_states(n_states)

    properties = compute_train_state_properties(
        panel,
        raw_states,
        train_indices,
        n_states=n_states,
        stress_score_components=stress_score_components,
    )

    scored = compute_state_stress_scores(
        properties,
        stress_score_components=stress_score_components,
    )

    raw_to_name, raw_to_id, monotonicity_passed, reason = _mapping_from_scores(
        scored,
        n_states=n_states,
    )

    scored["economic_state_name"] = scored["raw_state"].map(raw_to_name)
    scored["economic_state_id"] = scored["raw_state"].map(raw_to_id)
    scored["transition_state_modelled"] = bool(n_states == 3)
    scored["economic_monotonicity_passed"] = bool(monotonicity_passed)
    scored["state_labeling_rejection_reason"] = reason

    return HMMStateLabelingResult(
        n_states=n_states,
        raw_state_to_name=raw_to_name,
        raw_state_to_id=raw_to_id,
        transition_state_modelled=bool(n_states == 3),
        economic_monotonicity_passed=bool(monotonicity_passed),
        rejection_reason=reason,
        state_properties=scored,
    )


def map_raw_states_to_economic_names(
    raw_states: Sequence[int] | np.ndarray,
    labeling: HMMStateLabelingResult,
) -> np.ndarray:
    """Map raw HMM states to economic state names."""
    states = _validate_raw_states(raw_states, n_states=labeling.n_states)
    return np.array([labeling.raw_state_to_name[int(state)] for state in states], dtype=object)


def map_raw_states_to_economic_ids(
    raw_states: Sequence[int] | np.ndarray,
    labeling: HMMStateLabelingResult,
) -> np.ndarray:
    """Map raw HMM states to economic state ids."""
    states = _validate_raw_states(raw_states, n_states=labeling.n_states)
    return np.array([labeling.raw_state_to_id[int(state)] for state in states], dtype=int)


def append_hmm_state_labels(
    panel: pd.DataFrame,
    raw_states: Sequence[int] | np.ndarray,
    labeling: HMMStateLabelingResult,
    *,
    raw_state_col: str = HMM_RAW_STATE_COL,
    state_col: str = HMM_STATE_COL,
    state_name_col: str = HMM_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Append raw and economic state labels to a panel.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    states = _validate_raw_states(raw_states, n_states=labeling.n_states)

    if len(panel) != len(states):
        raise ValueError(f"panel/raw_states length mismatch: {len(panel)} vs {len(states)}.")

    out = panel.copy().reset_index(drop=True)

    out[raw_state_col] = states
    out[state_col] = map_raw_states_to_economic_ids(states, labeling)
    out[state_name_col] = map_raw_states_to_economic_names(states, labeling)

    score_map = (
        labeling.state_properties
        .set_index("raw_state")["hmm_state_stress_score"]
        .to_dict()
    )

    out[HMM_STATE_STRESS_SCORE_COL] = out[raw_state_col].map(score_map).astype(float)
    out[HMM_TRANSITION_STATE_MODELLED_COLUMN] = bool(labeling.transition_state_modelled)
    out[HMM_STATE_MAPPING_VALID_COL] = bool(labeling.economic_monotonicity_passed)
    out[HMM_STATE_MAPPING_REASON_COL] = labeling.rejection_reason

    return out


def map_raw_probabilities_to_economic_probabilities(
    raw_filtered_probs: np.ndarray,
    labeling: HMMStateLabelingResult,
) -> pd.DataFrame:
    """
    Convert raw-state filtered probabilities into economic-state probabilities.

    For K=2:
    - transition probability is set to 0.0
    - transition_state_modelled is False
    """
    probs = np.asarray(raw_filtered_probs, dtype=float)

    if probs.ndim != 2:
        raise ValueError(f"raw_filtered_probs must be 2D. Got shape {probs.shape}.")

    if probs.shape[1] != labeling.n_states:
        raise ValueError(
            f"raw_filtered_probs state dimension mismatch: "
            f"{probs.shape[1]} vs labeling.n_states={labeling.n_states}."
        )

    if not np.all(np.isfinite(probs)):
        raise ValueError("raw_filtered_probs contains non-finite values.")

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1.0e-8):
        raise ValueError(
            f"raw_filtered_probs rows must sum to 1. "
            f"min={row_sums.min()}, max={row_sums.max()}."
        )

    out = pd.DataFrame(
        {
            HMM_FILTERED_ECONOMIC_PROB_COLUMNS["calm"]: np.zeros(probs.shape[0]),
            HMM_FILTERED_ECONOMIC_PROB_COLUMNS["transition"]: np.zeros(probs.shape[0]),
            HMM_FILTERED_ECONOMIC_PROB_COLUMNS["stress"]: np.zeros(probs.shape[0]),
        }
    )

    for raw_state, economic_name in labeling.raw_state_to_name.items():
        col = HMM_FILTERED_ECONOMIC_PROB_COLUMNS[economic_name]
        out[col] += probs[:, int(raw_state)]

    if not labeling.transition_state_modelled:
        out[HMM_FILTERED_ECONOMIC_PROB_COLUMNS["transition"]] = 0.0

    econ_sum = out.loc[:, get_hmm_filtered_economic_probability_columns()].sum(axis=1)
    if not np.allclose(econ_sum, 1.0, atol=1.0e-8):
        raise ValueError(
            f"Economic probability rows must sum to 1. "
            f"min={econ_sum.min()}, max={econ_sum.max()}."
        )

    out[HMM_TRANSITION_STATE_MODELLED_COLUMN] = bool(labeling.transition_state_modelled)

    return out


def append_hmm_economic_probabilities(
    panel: pd.DataFrame,
    raw_filtered_probs: np.ndarray,
    labeling: HMMStateLabelingResult,
) -> pd.DataFrame:
    """
    Append economic filtered probability columns to a panel.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    econ_probs = map_raw_probabilities_to_economic_probabilities(
        raw_filtered_probs,
        labeling,
    )

    if len(panel) != len(econ_probs):
        raise ValueError(f"panel/probability length mismatch: {len(panel)} vs {len(econ_probs)}.")

    out = panel.copy().reset_index(drop=True)

    for col in econ_probs.columns:
        out[col] = econ_probs[col].to_numpy()

    return out


def add_hmm_next_session_signal_columns(
    panel: pd.DataFrame,
    *,
    date_col: str = "date",
    state_col: str = HMM_STATE_COL,
    state_name_col: str = HMM_STATE_NAME_COL,
) -> pd.DataFrame:
    """
    Add t+1 usability alias columns.

    Signal convention:
    - HMM state/probability at date t uses date t close data.
    - It is available after date t close.
    - It may be used only from the next session.

    This function creates aliases on row t saying what should be used for the
    next session. It does not shift future data backward.
    """
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame.")

    required = [
        date_col,
        state_col,
        state_name_col,
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["calm"],
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["transition"],
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["stress"],
    ]

    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"Cannot add signal columns. Missing columns: {missing}")

    out = panel.copy().reset_index(drop=True)

    dates = pd.to_datetime(out[date_col], errors="coerce")
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(f"{date_col!r} contains {bad_count} invalid date(s).")

    out["hmm_signal_observation_date"] = dates.dt.date.astype(str)
    out["hmm_signal_available_after_close_date"] = dates.dt.date.astype(str)

    if len(out) > 1:
        trade_dates = dates.shift(-1)
        out["hmm_signal_trade_date"] = trade_dates.dt.date.astype("string")
    else:
        out["hmm_signal_trade_date"] = pd.Series([pd.NA], dtype="string")

    out["hmm_state_for_next_session"] = out[state_col]
    out["hmm_state_name_for_next_session"] = out[state_name_col]

    out["hmm_filtered_prob_calm_for_next_session"] = out[
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["calm"]
    ]

    out["hmm_filtered_prob_transition_for_next_session"] = out[
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["transition"]
    ]

    out["hmm_filtered_prob_stress_for_next_session"] = out[
        HMM_FILTERED_ECONOMIC_PROB_COLUMNS["stress"]
    ]

    # Last row has no known next session in the current panel.
    if len(out) > 0:
        out.loc[len(out) - 1, "hmm_signal_trade_date"] = pd.NA

    for col in HMM_SIGNAL_AVAILABILITY_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Failed to create required signal column: {col}")

    return out


def assert_state_labeling_uses_train_only_inputs(
    train_indices: Sequence[int] | np.ndarray,
    *,
    n_rows: int,
) -> None:
    """
    Guardrail: train indices must be a strict prefix of the available panel.

    This prevents accidental use of full-sample state properties for mapping.
    """
    idx = _validate_train_indices(train_indices, n_rows=n_rows)
    expected = np.arange(0, len(idx), dtype=int)

    if not np.array_equal(idx, expected):
        raise ValueError(
            "State labeling train_indices must be the chronological train prefix."
        )

    if len(idx) >= n_rows:
        raise ValueError(
            "State labeling cannot use the full panel as train period."
        )


def state_labeling_result_to_frame(
    labeling: HMMStateLabelingResult,
) -> pd.DataFrame:
    """Return state labeling properties as a report-ready DataFrame."""
    out = labeling.state_properties.copy()

    required = [
        "raw_state",
        "economic_state_id",
        "economic_state_name",
        "hmm_state_stress_score",
        "n_train_observations",
        "train_occupancy",
        "transition_state_modelled",
        "economic_monotonicity_passed",
        "state_labeling_rejection_reason",
    ]

    for col in required:
        if col not in out.columns:
            out[col] = np.nan

    return out


__all__ = [
    "HMM_RAW_STATE_COL",
    "HMM_STATE_COL",
    "HMM_STATE_NAME_COL",
    "HMM_STATE_STRESS_SCORE_COL",
    "HMM_STATE_MAPPING_VALID_COL",
    "HMM_STATE_MAPPING_REASON_COL",
    "HMM_ECONOMIC_STATE_TO_ID",
    "HMM_ECONOMIC_ID_TO_STATE",
    "HMM_DEFAULT_STRESS_SCORE_COMPONENTS",
    "HMMStateLabelingResult",
    "compute_train_state_properties",
    "compute_state_stress_scores",
    "label_hmm_states_from_train_properties",
    "map_raw_states_to_economic_names",
    "map_raw_states_to_economic_ids",
    "append_hmm_state_labels",
    "map_raw_probabilities_to_economic_probabilities",
    "append_hmm_economic_probabilities",
    "add_hmm_next_session_signal_columns",
    "assert_state_labeling_uses_train_only_inputs",
    "state_labeling_result_to_frame",
]