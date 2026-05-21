from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_msgarch_inputs.py"


def load_export_module():
    spec = importlib.util.spec_from_file_location("export_msgarch_inputs", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def export_module():
    return load_export_module()


def write_test_config(tmp_path: Path, min_observations: int = 2) -> Path:
    cfg = {
        "model_name": "msgarch_appendix_v1",
        "implementation": "R_MSGARCH",
        "optional_phase": True,
        "required_for_main_pipeline": False,
        "input_policy": {
            "primary_input": "index_log_return",
            "prefilter_mean": True,
            "prefilter_method": "AR1",
            "return_scale": "percent",
            "min_observations": min_observations,
        },
        "markets": {
            "US": {
                "source_panel": "data/processed/us_vrp_har.parquet",
                "output_input_csv": "data/interim/msgarch/us_msgarch_input.csv",
            }
        },
        "return_column_candidates": ["log_return", "index_return", "simple_return"],
        "price_column_candidates": ["close", "adj_close", "underlying_close"],
        "output_policy": {
            "input_summary_csv": "reports/tables/phase_8/msgarch_input_summary.csv"
        },
    }

    path = tmp_path / "configs" / "model_msgarch.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return path


def write_source_panel(tmp_path: Path, df: pd.DataFrame) -> Path:
    path = tmp_path / "data" / "processed" / "us_vrp_har.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def test_log_return_has_priority_over_simple_return(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "log_return": [0.01, 0.02, -0.01],
                "simple_return": [0.50, 0.50, 0.50],
                "close": [100.0, 101.0, 102.0],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)
    result = export_module.export_market("US", config, tmp_path)

    out = pd.read_csv(result.output_csv)

    assert result.source_return_column == "log_return"
    assert result.derivation_method == "direct:log_return"
    assert out["return_for_msgarch"].tolist() == pytest.approx([1.0, 2.0, -1.0])


def test_simple_return_converts_to_log_return(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "simple_return": [0.01, 0.02, -0.01],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)
    result = export_module.export_market("US", config, tmp_path)

    out = pd.read_csv(result.output_csv)

    assert result.source_return_column == "simple_return"
    assert result.derivation_method == "log1p(simple_return)"
    assert out["log_return"].tolist() == pytest.approx(
        [
            0.009950330853168082,
            0.01980262729617973,
            -0.01005033585350145,
        ]
    )


def test_price_column_builds_diff_log_price(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4),
                "close": [100.0, 101.0, 103.0, 102.0],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)
    result = export_module.export_market("US", config, tmp_path)

    out = pd.read_csv(result.output_csv)

    assert result.source_return_column == "price:close"
    assert result.derivation_method == "diff(log(close))"
    assert len(out) == 3
    assert out["log_return"].tolist() == pytest.approx(
        [
            0.009950330853168082,
            0.019608471388376337,
            -0.009756174945364656,
        ]
    )


def test_missing_return_and_price_columns_fail(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "some_other_column": [1, 2, 3],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)

    with pytest.raises(export_module.MSGARCHExportError, match="Could not build MSGARCH input return"):
        export_module.export_market("US", config, tmp_path)


def test_simple_return_less_than_minus_one_fails(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "simple_return": [0.01, -1.01, 0.02],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)

    with pytest.raises(export_module.MSGARCHExportError, match="log1p"):
        export_module.export_market("US", config, tmp_path)


def test_output_schema_and_summary_are_written(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "log_return": [0.01, 0.02, -0.01],
            }
        ),
    )

    results, summary_path = export_module.run_export(
        market_arg="US",
        config_path=config_path,
        project_root=tmp_path,
    )

    assert len(results) == 1
    assert summary_path.exists()

    out = pd.read_csv(results[0].output_csv)
    assert list(out.columns) == [
        "date",
        "market",
        "log_return",
        "return_for_msgarch",
        "source_return_column",
        "input_available",
    ]

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "market"] == "US"
    assert summary.loc[0, "validation_status"] == "ok"
    assert summary.loc[0, "n_export_rows"] == 3
    assert isinstance(summary.loc[0, "input_hash_sha256"], str)
    assert len(str(summary.loc[0, "input_hash_sha256"])) == 64


def test_insufficient_observations_fail(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path, min_observations=5)
    write_source_panel(
        tmp_path,
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3),
                "log_return": [0.01, 0.02, -0.01],
            }
        ),
    )

    config = export_module.load_msgarch_config(config_path, tmp_path)

    with pytest.raises(export_module.MSGARCHExportError, match="below min_observations"):
        export_module.export_market("US", config, tmp_path)


def test_dry_run_writes_summary_without_source_file(tmp_path: Path, export_module):
    config_path = write_test_config(tmp_path, min_observations=5)

    results, summary_path = export_module.run_export(
        market_arg="US",
        config_path=config_path,
        project_root=tmp_path,
        dry_run=True,
    )

    assert len(results) == 1
    assert results[0].validation_status == "dry_run"
    assert summary_path.exists()

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "validation_status"] == "dry_run"