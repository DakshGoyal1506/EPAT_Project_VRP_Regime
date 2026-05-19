from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from vrp.regimes.hmm_features import (  # noqa: E402
    build_hmm_feature_panel,
    get_feature_matrix,
)
from vrp.regimes.hmm_registry import (  # noqa: E402
    HMM_CANDIDATE_RANKING_COLUMNS,
    HMM_SIGNAL_AVAILABILITY_COLUMNS,
    get_hmm_diagnostic_smoothed_probability_columns,
    get_hmm_filtered_economic_probability_columns,
    get_hmm_filtered_raw_probability_columns,
)
from vrp.regimes.hmm_scaling import scale_hmm_feature_panel  # noqa: E402
from vrp.regimes.gaussian_hmm import (  # noqa: E402
    HMMCandidateSpec,
    HMMFitConfig,
    fit_and_build_hmm_candidate_output,
)
from vrp.reports.hmm_diagnostics import (  # noqa: E402
    build_candidate_ranking_table as build_report_candidate_ranking_table,
    build_hmm_state_summary_table,
    build_hmm_transition_matrix_table,
    build_hmm_state_duration_summary_table,
    build_hmm_state_by_year_table,
    build_hmm_metadata,
)


def _make_two_state_hmm_panel(
    *,
    block: int = 60,
    repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regime = np.tile(np.r_[np.zeros(block), np.ones(block)], repeats).astype(int)
    n = len(regime)

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "vrp_har_gk": np.where(
                regime == 0,
                rng.normal(0.04, 0.01, n),
                rng.normal(-0.02, 0.02, n),
            ),
            "rv_gk_22d_ann_lag1": np.where(
                regime == 0,
                rng.normal(0.10, 0.02, n),
                rng.normal(0.35, 0.05, n),
            ),
            "iv_ann": np.where(
                regime == 0,
                rng.normal(0.15, 0.02, n),
                rng.normal(0.45, 0.05, n),
            ),
            "log_return": np.where(
                regime == 0,
                rng.normal(0.001, 0.005, n),
                rng.normal(-0.002, 0.02, n),
            ),
            "har_forecast_available": True,
        }
    )


def _make_three_state_hmm_panel(
    *,
    block: int = 40,
    repeats: int = 10,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    regime = np.tile(
        np.r_[
            np.zeros(block),
            np.ones(block),
            np.full(block, 2),
        ],
        repeats,
    ).astype(int)

    n = len(regime)

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "vrp_har_gk": np.select(
                [regime == 0, regime == 1, regime == 2],
                [
                    rng.normal(0.05, 0.008, n),
                    rng.normal(0.015, 0.012, n),
                    rng.normal(-0.025, 0.020, n),
                ],
            ),
            "rv_gk_22d_ann_lag1": np.select(
                [regime == 0, regime == 1, regime == 2],
                [
                    rng.normal(0.10, 0.015, n),
                    rng.normal(0.22, 0.025, n),
                    rng.normal(0.40, 0.050, n),
                ],
            ),
            "iv_ann": np.select(
                [regime == 0, regime == 1, regime == 2],
                [
                    rng.normal(0.15, 0.015, n),
                    rng.normal(0.28, 0.025, n),
                    rng.normal(0.50, 0.050, n),
                ],
            ),
            "log_return": np.select(
                [regime == 0, regime == 1, regime == 2],
                [
                    rng.normal(0.0015, 0.004, n),
                    rng.normal(0.0000, 0.010, n),
                    rng.normal(-0.0025, 0.020, n),
                ],
            ),
            "har_forecast_available": True,
        }
    )


def _fit_output(
    *,
    df: pd.DataFrame,
    n_states: int,
    covariance_type: str = "diag",
    feature_set: str = "F3",
):
    feature_panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set=feature_set,
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    scaled = scale_hmm_feature_panel(
        feature_panel,
        train_fraction=0.70,
        min_train_observations=750,
        min_test_observations=250,
    )

    return fit_and_build_hmm_candidate_output(
        scaled,
        spec=HMMCandidateSpec(feature_set, n_states, covariance_type),
        fit_config=HMMFitConfig(
            n_init=2,
            n_iter=150,
            random_seed=42,
        ),
    )


def test_hmm_feature_panel_builds_f3_matrix() -> None:
    df = _make_two_state_hmm_panel()

    panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    assert panel.market == "US"
    assert panel.feature_set == "F3"
    assert panel.feature_cols == (
        "vrp_har_gk",
        "rv_gk_22d_ann_lag1",
        "iv_ann",
        "index_return",
    )

    X = get_feature_matrix(panel, as_numpy=True)

    assert X.shape[0] == len(panel.eligible_panel)
    assert X.shape[1] == 4
    assert np.isfinite(X).all()


def test_hmm_feature_panel_blocks_unavailable_har_rows() -> None:
    df = _make_two_state_hmm_panel()
    df.loc[:9, "har_forecast_available"] = False

    panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=100,
        min_eligible_fraction=0.50,
    )

    assert len(panel.eligible_panel) == len(df) - 10
    assert not panel.blocked_rows.empty
    assert panel.blocked_rows["hmm_feature_blocked_reason"].str.contains(
        "condition_false:har_forecast_available"
    ).any()

    vrp_row = panel.availability_table[
        panel.availability_table["required_feature"] == "vrp_har_gk"
    ].iloc[0]

    assert vrp_row["required_condition"] == "har_forecast_available"
    assert int(vrp_row["n_condition_failed"]) == 10


def test_hmm_k2_output_schema_and_probability_rows() -> None:
    output = _fit_output(
        df=_make_two_state_hmm_panel(),
        n_states=2,
        covariance_type="diag",
    )

    assert output.output_panel is not None
    assert output.filter_result is not None
    assert output.raw_filtered_states is not None
    assert output.labeling is not None

    panel = output.output_panel

    raw_cols = get_hmm_filtered_raw_probability_columns(2)
    econ_cols = get_hmm_filtered_economic_probability_columns()
    smoothed_cols = get_hmm_diagnostic_smoothed_probability_columns(2)

    for col in raw_cols + econ_cols + smoothed_cols:
        assert col in panel.columns

    np.testing.assert_allclose(panel[raw_cols].sum(axis=1), 1.0, atol=1.0e-8)
    np.testing.assert_allclose(panel[econ_cols].sum(axis=1), 1.0, atol=1.0e-8)

    assert panel["hmm_filtered_prob_transition"].eq(0.0).all()
    assert panel["hmm_transition_state_modelled"].eq(False).all()

    for col in HMM_SIGNAL_AVAILABILITY_COLUMNS:
        assert col in panel.columns


def test_hmm_k3_output_schema_and_probability_rows() -> None:
    output = _fit_output(
        df=_make_three_state_hmm_panel(),
        n_states=3,
        covariance_type="diag",
    )

    assert output.output_panel is not None
    assert output.filter_result is not None
    assert output.raw_filtered_states is not None
    assert output.labeling is not None

    panel = output.output_panel

    raw_cols = get_hmm_filtered_raw_probability_columns(3)
    econ_cols = get_hmm_filtered_economic_probability_columns()
    smoothed_cols = get_hmm_diagnostic_smoothed_probability_columns(3)

    for col in raw_cols + econ_cols + smoothed_cols:
        assert col in panel.columns

    np.testing.assert_allclose(panel[raw_cols].sum(axis=1), 1.0, atol=1.0e-8)
    np.testing.assert_allclose(panel[econ_cols].sum(axis=1), 1.0, atol=1.0e-8)

    assert panel["hmm_transition_state_modelled"].eq(True).all()


def test_hmm_candidate_validation_fields_are_present() -> None:
    output = _fit_output(
        df=_make_two_state_hmm_panel(),
        n_states=2,
        covariance_type="diag",
    )

    assert output.validation.market == "US"
    assert output.validation.feature_set == "F3"
    assert output.validation.n_states == 2
    assert output.validation.covariance_type == "diag"

    assert isinstance(output.validation.hmm_model_valid, bool)
    assert isinstance(output.validation.hmm_model_failure_reason, str)
    assert output.validation.probability_validation is not None
    assert output.validation.train_occupancy is not None
    assert output.validation.test_occupancy is not None


def test_hmm_report_tables_build_from_candidate_output() -> None:
    output = _fit_output(
        df=_make_two_state_hmm_panel(),
        n_states=2,
        covariance_type="diag",
    )

    ranking = build_report_candidate_ranking_table([output], selected_output=output)
    summary = build_hmm_state_summary_table(output)
    transition = build_hmm_transition_matrix_table(output)
    duration = build_hmm_state_duration_summary_table(output)
    by_year = build_hmm_state_by_year_table(output)
    metadata = build_hmm_metadata(output)

    assert list(ranking.columns) == list(HMM_CANDIDATE_RANKING_COLUMNS)
    assert not summary.empty
    assert not transition.empty
    assert not duration.empty
    assert not by_year.empty

    assert metadata["market"] == "US"
    assert metadata["feature_set"] == "F3"
    assert metadata["n_states"] == 2
    assert "train_window_hash" in metadata
    assert "scaler_hash" in metadata
    assert "hmm_parameter_hash" in metadata


def test_hmm_candidate_ranking_marks_selected_output() -> None:
    output = _fit_output(
        df=_make_two_state_hmm_panel(),
        n_states=2,
        covariance_type="diag",
    )

    ranking = build_report_candidate_ranking_table([output], selected_output=output)

    assert len(ranking) == 1
    assert bool(ranking.loc[0, "selected_primary"])
    assert ranking.loc[0, "feature_set"] == "F3"
    # ensure we convert any numpy scalar to a native int for type-checkers
    assert int(ranking.loc[0, "n_states"].item()) == 2
    assert ranking.loc[0, "covariance_type"] == "diag"


def test_hmm_model_failure_path_for_bad_feature_availability() -> None:
    df = _make_two_state_hmm_panel()
    df["har_forecast_available"] = False

    panel = build_hmm_feature_panel(
        df,
        market="US",
        feature_set="F3",
        min_eligible_observations=1000,
        min_eligible_fraction=0.50,
    )

    assert panel.eligible_panel.empty
    assert not panel.availability_summary.passed
    assert "feature_availability_too_low" in panel.availability_summary.blocked_reason