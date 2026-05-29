"""
No-live-order source guard for Phase 11.

Phase 11 is paper-signal / paper-intent only. It must not contain executable
broker order-submission code.

This module scans Python source using AST. It intentionally avoids raw string
matching because documentation strings may mention forbidden APIs while clearly
stating that they are not used.

It blocks:
- direct imports of broker execution libraries
- function definitions with execution names
- direct calls to execution functions
- attribute calls such as client.placeOrder(...)
- getattr(client, "placeOrder") style access
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class LiveOrderGuardError(RuntimeError):
    """Raised when live-order code is detected."""


FORBIDDEN_EXECUTION_SYMBOLS = frozenset(
    {
        "placeOrder",
        "place_order",
        "submitOrder",
        "submit_order",
        "sendOrder",
        "send_order",
        "order_target",
        "order_value",
        "order_percent",
        "market_order",
        "limit_order",
        "stop_order",
        "bracket_order",
        "cancelOrder",
        "cancel_order",
        "reqGlobalCancel",
        "buy",
        "sell",
    }
)

FORBIDDEN_DIRECT_IMPORT_ROOTS = frozenset(
    {
        "ib_insync",
        "IBridgePy",
        "ibridgepy",
    }
)

DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "htmlcov",
        "data",
        "logs",
        "reports",
        "notebooks",
    }
)


@dataclass(frozen=True)
class LiveOrderGuardViolation:
    """One no-live-order guard violation."""

    path: str
    line: int
    column: int
    violation_type: str
    symbol: str
    context: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "violation_type": self.violation_type,
            "symbol": self.symbol,
            "context": self.context,
        }


@dataclass(frozen=True)
class LiveOrderGuardReport:
    """No-live-order guard scan result."""

    scanned_paths: tuple[str, ...]
    violations: tuple[LiveOrderGuardViolation, ...]

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    @property
    def blocked_reason(self) -> str:
        if self.passed:
            return "no live-order source violations detected"

        return f"{len(self.violations)} live-order source violation(s) detected"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocked_reason": self.blocked_reason,
            "scanned_paths": list(self.scanned_paths),
            "violations": [violation.as_dict() for violation in self.violations],
        }


class _LiveOrderAstVisitor(ast.NodeVisitor):
    """AST visitor that records executable live-order patterns."""

    def __init__(self, *, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.violations: list[LiveOrderGuardViolation] = []
        self._lines = source.splitlines()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", maxsplit=1)[0]
            if root in FORBIDDEN_DIRECT_IMPORT_ROOTS:
                self._add_violation(
                    node=node,
                    violation_type="forbidden_direct_import",
                    symbol=alias.name,
                )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        root = module.split(".", maxsplit=1)[0]

        if root in FORBIDDEN_DIRECT_IMPORT_ROOTS:
            self._add_violation(
                node=node,
                violation_type="forbidden_direct_import",
                symbol=module,
            )

        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name in FORBIDDEN_EXECUTION_SYMBOLS:
            self._add_violation(
                node=node,
                violation_type="forbidden_function_definition",
                symbol=node.name,
            )

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name in FORBIDDEN_EXECUTION_SYMBOLS:
            self._add_violation(
                node=node,
                violation_type="forbidden_async_function_definition",
                symbol=node.name,
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callable_name = _callable_name(node.func)

        if callable_name in FORBIDDEN_EXECUTION_SYMBOLS:
            self._add_violation(
                node=node,
                violation_type="forbidden_execution_call",
                symbol=callable_name,
            )

        if callable_name == "getattr" and len(node.args) >= 2:
            attr_name = _constant_string(node.args[1])
            if attr_name in FORBIDDEN_EXECUTION_SYMBOLS:
                self._add_violation(
                    node=node,
                    violation_type="forbidden_execution_getattr",
                    symbol=attr_name,
                )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_EXECUTION_SYMBOLS:
            self._add_violation(
                node=node,
                violation_type="forbidden_execution_attribute_reference",
                symbol=node.attr,
            )

        self.generic_visit(node)

    def _add_violation(
        self,
        *,
        node: ast.AST,
        violation_type: str,
        symbol: str,
    ) -> None:
        line = int(getattr(node, "lineno", 0) or 0)
        column = int(getattr(node, "col_offset", 0) or 0)
        context = self._context_for_line(line)

        self.violations.append(
            LiveOrderGuardViolation(
                path=str(self.path),
                line=line,
                column=column,
                violation_type=violation_type,
                symbol=symbol,
                context=context,
            )
        )

    def _context_for_line(self, line: int) -> str:
        if line <= 0:
            return ""

        index = line - 1
        if index >= len(self._lines):
            return ""

        return self._lines[index].strip()


def iter_python_files(
    roots: Iterable[str | Path],
    *,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> tuple[Path, ...]:
    """Return Python files under roots, excluding generated/data/report dirs."""

    excluded = set(excluded_dirs)
    files: list[Path] = []

    for root in roots:
        path = Path(root)

        if not path.exists():
            continue

        if path.is_file():
            if path.suffix == ".py":
                files.append(path)
            continue

        for candidate in path.rglob("*.py"):
            parts = set(candidate.parts)
            if parts.intersection(excluded):
                continue

            files.append(candidate)

    return tuple(sorted(files, key=lambda item: str(item)))


def scan_python_file(path: str | Path) -> tuple[LiveOrderGuardViolation, ...]:
    """Scan one Python file for live-order source violations."""

    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        violation = LiveOrderGuardViolation(
            path=str(source_path),
            line=int(exc.lineno or 0),
            column=int(exc.offset or 0),
            violation_type="syntax_error",
            symbol="syntax_error",
            context=str(exc),
        )
        return (violation,)

    visitor = _LiveOrderAstVisitor(path=source_path, source=source)
    visitor.visit(tree)

    return tuple(visitor.violations)


def scan_source_tree(
    roots: Iterable[str | Path],
    *,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> LiveOrderGuardReport:
    """Scan source tree for executable live-order code."""

    files = iter_python_files(roots, excluded_dirs=excluded_dirs)
    violations: list[LiveOrderGuardViolation] = []

    for file_path in files:
        violations.extend(scan_python_file(file_path))

    return LiveOrderGuardReport(
        scanned_paths=tuple(str(path) for path in files),
        violations=tuple(violations),
    )


def assert_no_live_order_code(
    roots: Iterable[str | Path],
    *,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> LiveOrderGuardReport:
    """Raise if executable live-order code is detected."""

    report = scan_source_tree(roots, excluded_dirs=excluded_dirs)

    if report.violations:
        preview = "\n".join(
            (
                f"- {violation.path}:{violation.line}:{violation.column} "
                f"{violation.violation_type} {violation.symbol!r} "
                f"-> {violation.context}"
            )
            for violation in report.violations[:20]
        )
        raise LiveOrderGuardError(
            f"{report.blocked_reason}\n{preview}"
        )

    return report


def write_live_order_guard_report(
    report: LiveOrderGuardReport,
    output_path: str | Path,
) -> Path:
    """Write no-live-order guard report to JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return path


def _callable_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr

    return None


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    return None
