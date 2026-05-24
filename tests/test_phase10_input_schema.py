from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrp.backtest.schema_audit import (
    APPROVED_PHASE9_STRATEGIES,
    MarketInputPaths,
    PRIMARY_PAYOFF_LABEL,
    audit_market_inputs,
    assert_no_audit_errors,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def _fixture_paths(
    *,
    signal_path: Path | None = None,
    outcome_path: Path | None = None,
) -> MarketInputPaths:
    signal_path = signal_path or FIXTURE_DIR / "phase10_signals_toy.csv"
    outcome_path = outcome_path or FIXTURE_DIR / "phase10_outcomes_toy.csv"

    return MarketInputPaths(
        market="US",
        strategy_signals=signal_path,
        vrp_har=outcome_path,
        vrp=outcome_path,
    )


def test_toy_phase10_inputs_pass_schema_audit() -> None:
    result = audit_market_inputs(_fixture_paths())

    assert not result.has_errors()
    assert result.error_count() == 0

    signal_schema_rows = [
        row for row in result.rows if row["component"] == "strategy_signals_schema"
    ]
    assert len(signal_schema_rows) == 1


def test_toy_strategy_universe_is_exactly_approved() -> None:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    strategies = set(signals["strategy_name"].unique())

    assert strategies == set(APPROVED_PHASE9_STRATEGIES)


def test_missing_signal_column_fails(tmp_path: Path) -> None:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    signals = signals.drop(columns=["target_trade_date"])

    bad_signal_path = tmp_path / "bad_signals.csv"
    signals.to_csv(bad_signal_path, index=False)

    result = audit_market_inputs(_fixture_paths(signal_path=bad_signal_path))

    assert result.has_errors()
    assert any(
        "Missing required signal columns" in issue.message
        for issue in result.issues
        if issue.severity == "error"
    )


def test_duplicate_signal_key_fails(tmp_path: Path) -> None:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    duplicated = pd.concat([signals, signals.iloc[[0]]], ignore_index=True)

    bad_signal_path = tmp_path / "duplicate_signals.csv"
    duplicated.to_csv(bad_signal_path, index=False)

    result = audit_market_inputs(_fixture_paths(signal_path=bad_signal_path))

    assert result.has_errors()
    assert any(
        "Duplicate signal rows found" in issue.message
        for issue in result.issues
        if issue.severity == "error"
    )


def test_extra_strategy_fails(tmp_path: Path) -> None:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    signals.loc[len(signals)] = {
        "market": "US",
        "strategy_name": "new_unapproved_strategy",
        "signal_observation_date": "2020-01-02",
        "target_trade_date": "2020-01-03",
        "target_exposure": -1.0,
        "strategy_available": True,
    }

    bad_signal_path = tmp_path / "extra_strategy_signals.csv"
    signals.to_csv(bad_signal_path, index=False)

    result = audit_market_inputs(_fixture_paths(signal_path=bad_signal_path))

    assert result.has_errors()
    assert any(
        "Strategy universe mismatch" in issue.message
        for issue in result.issues
        if issue.severity == "error"
    )


def test_msvol_strategy_fails(tmp_path: Path) -> None:
    signals = pd.read_csv(FIXTURE_DIR / "phase10_signals_toy.csv")
    signals.loc[0, "strategy_name"] = "msvol_appendix_strategy"

    bad_signal_path = tmp_path / "msvol_signals.csv"
    signals.to_csv(bad_signal_path, index=False)

    result = audit_market_inputs(_fixture_paths(signal_path=bad_signal_path))

    assert result.has_errors()
    assert any(
        "MSVOL strategy rows are forbidden" in issue.message
        for issue in result.issues
        if issue.severity == "error"
    )


def test_missing_outcome_label_fails_when_no_candidate_has_label(tmp_path: Path) -> None:
    outcomes = pd.read_csv(FIXTURE_DIR / "phase10_outcomes_toy.csv")
    outcomes = outcomes.drop(columns=[PRIMARY_PAYOFF_LABEL])

    bad_outcome_path = tmp_path / "bad_outcomes.csv"
    outcomes.to_csv(bad_outcome_path, index=False)

    result = audit_market_inputs(_fixture_paths(outcome_path=bad_outcome_path))

    assert result.has_errors()
    assert any(
        "No valid outcome candidate" in issue.message
        for issue in result.issues
        if issue.severity == "error"
    )


def test_assert_no_audit_errors_raises_on_errors(tmp_path: Path) -> None:
    missing_signal_path = tmp_path / "does_not_exist.csv"

    result = audit_market_inputs(_fixture_paths(signal_path=missing_signal_path))

    try:
        assert_no_audit_errors([result])
    except AssertionError as exc:
        assert "Phase 10 input audit failed" in str(exc)
    else:
        raise AssertionError("Expected assert_no_audit_errors to raise.")