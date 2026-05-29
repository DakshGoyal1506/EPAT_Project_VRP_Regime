from __future__ import annotations

import json
from pathlib import Path

import pytest

from vrp.broker.live_order_guard import (
    LiveOrderGuardError,
    assert_no_live_order_code,
    iter_python_files,
    scan_python_file,
    scan_source_tree,
    write_live_order_guard_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scan_clean_file_passes(tmp_path: Path) -> None:
    path = tmp_path / "clean.py"
    path.write_text(
        """
def build_paper_intent():
    return {"live_order_sent": False}
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert violations == ()


def test_scan_ignores_docstring_and_string_mentions(tmp_path: Path) -> None:
    path = tmp_path / "safe_mentions.py"
    path.write_text(
        '''
"""
This file mentions placeOrder in documentation but does not call it.
"""

def explain():
    return "Do not call placeOrder in Phase 11."
''',
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert violations == ()


def test_scan_detects_place_order_call(tmp_path: Path) -> None:
    path = tmp_path / "bad_call.py"
    path.write_text(
        """
def bad(client, contract, order):
    client.placeOrder(1, contract, order)
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert any(v.symbol == "placeOrder" for v in violations)
    assert any(v.violation_type == "forbidden_execution_call" for v in violations)


def test_scan_detects_function_definition(tmp_path: Path) -> None:
    path = tmp_path / "bad_def.py"
    path.write_text(
        """
def place_order(order):
    return order
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert len(violations) == 1
    assert violations[0].symbol == "place_order"
    assert violations[0].violation_type == "forbidden_function_definition"


def test_scan_detects_getattr_execution_access(tmp_path: Path) -> None:
    path = tmp_path / "bad_getattr.py"
    path.write_text(
        """
def bad(client):
    fn = getattr(client, "placeOrder")
    return fn
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert any(v.symbol == "placeOrder" for v in violations)
    assert any(v.violation_type == "forbidden_execution_getattr" for v in violations)


def test_scan_detects_direct_ib_insync_import(tmp_path: Path) -> None:
    path = tmp_path / "bad_import.py"
    path.write_text(
        """
from ib_insync import IB
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert len(violations) == 1
    assert violations[0].symbol == "ib_insync"
    assert violations[0].violation_type == "forbidden_direct_import"


def test_scan_detects_direct_ibridgepy_import(tmp_path: Path) -> None:
    path = tmp_path / "bad_import.py"
    path.write_text(
        """
import IBridgePy
""",
        encoding="utf-8",
    )

    violations = scan_python_file(path)

    assert len(violations) == 1
    assert violations[0].symbol == "IBridgePy"
    assert violations[0].violation_type == "forbidden_direct_import"


def test_scan_source_tree_detects_bad_file(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    bad = tmp_path / "bad.py"

    good.write_text(
        """
def paper_only():
    return False
""",
        encoding="utf-8",
    )
    bad.write_text(
        """
def bad(client):
    client.submit_order("SPY", 1)
""",
        encoding="utf-8",
    )

    report = scan_source_tree([tmp_path])

    assert report.passed is False
    assert len(report.violations) >= 1
    assert any(v.symbol == "submit_order" for v in report.violations)


def test_assert_no_live_order_code_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text(
        """
def bad(client):
    client.buy("SPY", 1)
""",
        encoding="utf-8",
    )

    with pytest.raises(LiveOrderGuardError, match="live-order source violation"):
        assert_no_live_order_code([tmp_path])


def test_iter_python_files_excludes_reports_and_data(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"

    src_dir.mkdir()
    data_dir.mkdir()
    reports_dir.mkdir()

    src_file = src_dir / "x.py"
    data_file = data_dir / "x.py"
    reports_file = reports_dir / "x.py"

    src_file.write_text("x = 1\n", encoding="utf-8")
    data_file.write_text("x = 1\n", encoding="utf-8")
    reports_file.write_text("x = 1\n", encoding="utf-8")

    files = iter_python_files([tmp_path])

    assert src_file in files
    assert data_file not in files
    assert reports_file not in files


def test_write_live_order_guard_report(tmp_path: Path) -> None:
    path = tmp_path / "clean.py"
    path.write_text("x = 1\n", encoding="utf-8")

    report = scan_source_tree([tmp_path])
    output_path = tmp_path / "guard_report.json"

    written = write_live_order_guard_report(report, output_path)

    assert written == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["violations"] == []


def test_repository_src_and_scripts_have_no_live_order_code() -> None:
    report = assert_no_live_order_code(
        [
            REPO_ROOT / "src",
            REPO_ROOT / "scripts",
        ]
    )

    assert report.passed is True
    assert report.violations == ()
    assert any("src" in path for path in report.scanned_paths)
    assert any("scripts" in path for path in report.scanned_paths)