from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from vrp.broker.signal_schema import (
    Phase9SignalSchema,
    SignalSchemaError,
    canonicalize_signal_frame,
    check_signal_freshness,
    latest_signal_to_record,
    load_signal_frame,
    load_validate_select_latest_signal,
    select_latest_signal,
    validate_signal_schema,
)


def _valid_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_name": [
                "mar_prob_linear_carry",
                "mar_prob_linear_carry",
                "mar_prob_linear_carry",
            ],
            "signal_observation_date": [
                "2026-05-08",
                "2026-05-09",
                "2026-05-10",
            ],
            "target_trade_date": [
                "2026-05-11",
                "2026-05-12",
                "2026-05-13",
            ],
            "target_exposure": [-0.25, -0.50, 0.0],
            "strategy_available": [True, True, True],
            "blocked_reason": ["", "", ""],
            "decision_reason": [
                "carry gate valid",
                "carry gate valid",
                "flat signal",
            ],
            "vrp_har_gk": [0.02, 0.03, 0.01],
            "har_forecast_available": [True, True, True],
            "state_name": ["calm", "calm", "transition"],
            "p_calm": [0.7, 0.8, 0.4],
            "p_transition": [0.2, 0.1, 0.5],
            "p_stress": [0.1, 0.1, 0.1],
        }
    )


def test_canonicalize_signal_frame_accepts_strategy_alias() -> None:
    df = _valid_signal_frame().rename(columns={"strategy_name": "strategy"})

    out = canonicalize_signal_frame(df)

    assert "strategy_name" in out.columns
    assert "strategy" not in out.columns
    assert out["strategy_name"].iloc[0] == "mar_prob_linear_carry"


def test_canonicalize_signal_frame_accepts_signal_date_alias() -> None:
    df = _valid_signal_frame().rename(
        columns={"signal_observation_date": "signal_date"}
    )

    out = canonicalize_signal_frame(df)

    assert "signal_observation_date" in out.columns
    assert "signal_date" not in out.columns


def test_validate_signal_schema_accepts_valid_frame() -> None:
    out = validate_signal_schema(_valid_signal_frame())

    assert len(out) == 3
    assert out["target_trade_date"].iloc[0] == date(2026, 5, 11)
    assert out["signal_observation_date"].iloc[0] == date(2026, 5, 8)
    assert out["target_exposure"].dtype == float
    assert out["strategy_available"].dtype == bool


def test_validate_signal_schema_accepts_boolean_like_values() -> None:
    df = _valid_signal_frame()
    df["strategy_available"] = ["true", "1", "yes"]

    out = validate_signal_schema(df)

    assert out["strategy_available"].tolist() == [True, True, True]


def test_validate_signal_schema_rejects_missing_strategy_column() -> None:
    df = _valid_signal_frame().drop(columns=["strategy_name"])

    with pytest.raises(SignalSchemaError, match="strategy_name or strategy"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_missing_required_column() -> None:
    df = _valid_signal_frame().drop(columns=["target_exposure"])

    with pytest.raises(SignalSchemaError, match="missing required columns"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_unapproved_strategy() -> None:
    df = _valid_signal_frame()
    df.loc[0, "strategy_name"] = "msvol"

    with pytest.raises(SignalSchemaError, match="outside approved universe"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_bad_date() -> None:
    df = _valid_signal_frame()
    df.loc[0, "target_trade_date"] = "not-a-date"

    with pytest.raises(SignalSchemaError, match="unparseable dates"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_non_numeric_exposure() -> None:
    df = _valid_signal_frame()
    df["target_exposure"] = df["target_exposure"].astype(object)
    df.loc[0, "target_exposure"] = "bad-exposure"

    with pytest.raises(SignalSchemaError, match="non-numeric"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_positive_exposure() -> None:
    df = _valid_signal_frame()
    df.loc[0, "target_exposure"] = 0.25

    with pytest.raises(SignalSchemaError, match="out of allowed range"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_too_negative_exposure() -> None:
    df = _valid_signal_frame()
    df.loc[0, "target_exposure"] = -1.25

    with pytest.raises(SignalSchemaError, match="out of allowed range"):
        validate_signal_schema(df)


def test_validate_signal_schema_rejects_bad_boolean() -> None:
    df = _valid_signal_frame()
    df["strategy_available"] = df["strategy_available"].astype(object)
    df.loc[0, "strategy_available"] = "maybe"

    with pytest.raises(SignalSchemaError, match="non-boolean"):
        validate_signal_schema(df)


def test_validate_signal_schema_allows_unavailable_rows_with_nat_target_date() -> None:
    df = _valid_signal_frame()

    unavailable = pd.DataFrame(
        {
            "strategy_name": ["mar_prob_linear_carry"],
            "signal_observation_date": ["2026-05-14"],
            "target_trade_date": [pd.NaT],
            "target_exposure": [float("nan")],
            "strategy_available": [False],
            "blocked_reason": ["missing_target_trade_date"],
            "decision_reason": ["unavailable"],
        }
    )

    df = pd.concat([df, unavailable], ignore_index=True)

    out = validate_signal_schema(df)

    assert len(out) == 4
    assert pd.isna(out.iloc[-1]["target_trade_date"])
    assert pd.isna(out.iloc[-1]["target_exposure"])
    assert out.iloc[-1]["strategy_available"] is False or out.iloc[-1]["strategy_available"] == False


def test_select_latest_signal_uses_target_trade_date_not_row_order() -> None:
    df = _valid_signal_frame().iloc[[2, 0, 1]].reset_index(drop=True)

    latest = select_latest_signal(df, strategy_name="mar_prob_linear_carry")

    assert latest["target_trade_date"] == date(2026, 5, 13)
    assert latest["target_exposure"] == 0.0


def test_select_latest_signal_filters_strategy_name() -> None:
    df = pd.concat(
        [
            _valid_signal_frame(),
            pd.DataFrame(
                {
                    "strategy_name": ["hmm_prob_linear_carry"],
                    "signal_observation_date": ["2026-05-14"],
                    "target_trade_date": ["2026-05-15"],
                    "target_exposure": [-0.75],
                    "strategy_available": [True],
                    "blocked_reason": [""],
                    "decision_reason": ["hmm carry valid"],
                }
            ),
        ],
        ignore_index=True,
    )

    latest = select_latest_signal(df, strategy_name="hmm_prob_linear_carry")

    assert latest["strategy_name"] == "hmm_prob_linear_carry"
    assert latest["target_trade_date"] == date(2026, 5, 15)


def test_select_latest_signal_filters_market_if_column_exists() -> None:
    df = pd.concat(
        [
            _valid_signal_frame().assign(market="US"),
            _valid_signal_frame().assign(
                market="INDIA",
                target_trade_date=[
                    "2026-05-14",
                    "2026-05-15",
                    "2026-05-16",
                ],
            ),
        ],
        ignore_index=True,
    )

    latest = select_latest_signal(
        df,
        strategy_name="mar_prob_linear_carry",
        market="US",
    )

    assert latest["market"] == "US"
    assert latest["target_trade_date"] == date(2026, 5, 13)


def test_select_latest_signal_rejects_unknown_strategy_request() -> None:
    with pytest.raises(SignalSchemaError, match="not approved"):
        select_latest_signal(_valid_signal_frame(), strategy_name="bad_strategy")


def test_select_latest_signal_rejects_empty_filter() -> None:
    df = _valid_signal_frame()

    with pytest.raises(SignalSchemaError, match="No Phase 9 signal rows found"):
        select_latest_signal(df, strategy_name="hmm_prob_linear")


def test_select_latest_signal_skips_unavailable_nat_target_date() -> None:
    df = _valid_signal_frame()

    unavailable = pd.DataFrame(
        {
            "strategy_name": ["mar_prob_linear_carry"],
            "signal_observation_date": ["2026-05-14"],
            "target_trade_date": [pd.NaT],
            "target_exposure": [float("nan")],
            "strategy_available": [False],
            "blocked_reason": ["missing_target_trade_date"],
            "decision_reason": ["unavailable"],
        }
    )

    df = pd.concat([df, unavailable], ignore_index=True)

    latest = select_latest_signal(df, strategy_name="mar_prob_linear_carry")

    assert latest["target_trade_date"] == date(2026, 5, 13)
    assert latest["target_exposure"] == 0.0


def test_check_signal_freshness_accepts_recent_signal() -> None:
    latest = {
        "target_trade_date": date(2026, 5, 13),
    }

    result = check_signal_freshness(
        latest,
        as_of_date=date(2026, 5, 15),
        max_signal_age_days=5,
    )

    assert result.is_stale is False
    assert result.age_days == 2
    assert result.final_status_if_blocked is None


def test_check_signal_freshness_blocks_stale_signal() -> None:
    latest = {
        "target_trade_date": date(2026, 5, 1),
    }

    result = check_signal_freshness(
        latest,
        as_of_date=date(2026, 5, 15),
        max_signal_age_days=5,
        block_stale_signal=True,
    )

    assert result.is_stale is True
    assert result.age_days == 14
    assert result.final_status_if_blocked == "BLOCKED_STALE_SIGNAL"


def test_check_signal_freshness_weekend_gap_minimum() -> None:
    latest = {
        "target_trade_date": date(2026, 5, 8),
    }

    result = check_signal_freshness(
        latest,
        as_of_date=date(2026, 5, 11),
        max_signal_age_days=1,
        allow_weekend_gap=True,
    )

    assert result.is_stale is False
    assert result.age_days == 3


def test_check_signal_freshness_without_weekend_gap_blocks() -> None:
    latest = {
        "target_trade_date": date(2026, 5, 8),
    }

    result = check_signal_freshness(
        latest,
        as_of_date=date(2026, 5, 11),
        max_signal_age_days=1,
        allow_weekend_gap=False,
    )

    assert result.is_stale is True
    assert result.final_status_if_blocked == "BLOCKED_STALE_SIGNAL"


def test_load_signal_frame_accepts_csv_for_tests(tmp_path: Path) -> None:
    path = tmp_path / "signals.csv"
    _valid_signal_frame().to_csv(path, index=False)

    out = load_signal_frame(path)

    assert len(out) == 3


def test_load_signal_frame_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(SignalSchemaError, match="Signal file not found"):
        load_signal_frame(tmp_path / "missing.parquet")


def test_load_signal_frame_rejects_bad_extension(tmp_path: Path) -> None:
    path = tmp_path / "signals.txt"
    path.write_text("bad", encoding="utf-8")

    with pytest.raises(SignalSchemaError, match="Unsupported signal file format"):
        load_signal_frame(path)


def test_load_validate_select_latest_signal_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "signals.csv"
    _valid_signal_frame().to_csv(path, index=False)

    latest = load_validate_select_latest_signal(
        path,
        strategy_name="mar_prob_linear_carry",
    )

    assert latest["target_trade_date"] == date(2026, 5, 13)


def test_latest_signal_to_record_serializes_dates() -> None:
    latest = select_latest_signal(_valid_signal_frame())
    record = latest_signal_to_record(latest)

    assert record["target_trade_date"] == "2026-05-13"
    assert record["signal_observation_date"] == "2026-05-10"


def test_custom_schema_can_extend_exposure_range() -> None:
    schema = Phase9SignalSchema.default()
    schema = Phase9SignalSchema(
        required_columns=schema.required_columns,
        strategy_column_candidates=schema.strategy_column_candidates,
        optional_columns=schema.optional_columns,
        aliases=schema.aliases,
        target_exposure_min=-1.0,
        target_exposure_max=1.0,
        latest_signal_sort_column=schema.latest_signal_sort_column,
        approved_strategies=schema.approved_strategies,
    )

    df = _valid_signal_frame()
    df.loc[0, "target_exposure"] = 0.25

    out = validate_signal_schema(df, schema)

    assert out["target_exposure"].iloc[0] == 0.25