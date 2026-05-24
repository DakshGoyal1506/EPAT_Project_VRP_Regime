from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TRADABLE_PROXY_SEARCH_DIRS: tuple[str, ...] = (
    "data/processed",
    "data/interim",
    "data/raw",
    "data/broker_cache",
)

TRADABLE_PROXY_NAME_TOKENS: tuple[str, ...] = (
    "vxx",
    "svxy",
    "uvxy",
    "vix_future",
    "vix_futures",
    "vx_future",
    "vx_futures",
    "option_proxy",
    "options_proxy",
    "option_chain",
    "options_chain",
    "tradable_proxy",
    "vol_etp",
    "volatility_etp",
)

SUPPORTED_PROXY_FILE_SUFFIXES: tuple[str, ...] = (
    ".parquet",
    ".csv",
    ".json",
)


@dataclass(frozen=True)
class TradableProxyCandidate:
    path: str
    suffix: str
    matched_token: str


@dataclass(frozen=True)
class TradableProxyDetectionResult:
    status: str
    reason: str
    n_candidates: int
    candidates: list[TradableProxyCandidate]
    generated_at_utc: str


class TradableProxyDetectionError(ValueError):
    """Raised when tradable-proxy detection cannot be completed."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))

    return value


def detect_tradable_proxy_data(
    repo_root: Path,
    *,
    search_dirs: tuple[str, ...] = TRADABLE_PROXY_SEARCH_DIRS,
    name_tokens: tuple[str, ...] = TRADABLE_PROXY_NAME_TOKENS,
) -> TradableProxyDetectionResult:
    """
    Detect existing tradable proxy files only.

    This does not download anything. It only scans local repo directories for
    already-present VIX ETP, VIX futures, option proxy, or related files.
    """
    repo_root = Path(repo_root)

    candidates: list[TradableProxyCandidate] = []

    for relative_dir in search_dirs:
        root = repo_root / relative_dir
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_PROXY_FILE_SUFFIXES:
                continue

            lower_name = path.name.lower()
            matched = [token for token in name_tokens if token in lower_name]
            if not matched:
                continue

            for token in matched:
                candidates.append(
                    TradableProxyCandidate(
                        path=str(path),
                        suffix=suffix,
                        matched_token=token,
                    )
                )

    candidates = sorted(candidates, key=lambda item: (item.path, item.matched_token))

    if candidates:
        return TradableProxyDetectionResult(
            status="available",
            reason="Existing tradable proxy candidate files were found. No downloads were performed.",
            n_candidates=len(candidates),
            candidates=candidates,
            generated_at_utc=datetime.now(UTC).isoformat(),
        )

    return TradableProxyDetectionResult(
        status="skipped",
        reason="required tradable proxy data not found; Phase 10 does not download new tradable proxy data",
        n_candidates=0,
        candidates=[],
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


def write_tradable_proxy_detection_report(
    result: TradableProxyDetectionResult,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return output_path