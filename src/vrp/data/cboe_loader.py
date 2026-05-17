"""CBOE VIX data loader.

The preferred path is:
1. Use local CSV if provided.
2. Otherwise try configured official URL.
3. If both fail, raise an actionable error telling the user to manually download
   the CBOE VIX CSV and set local_csv_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vrp.data.base import BaseDataLoader, DataIngestionError, DataLoadResult
from vrp.data.cleaners import standardize_single_close_series
from vrp.data.validators import validate_ohlcv_frame


def load_cboe_vix_from_csv(
    csv_path: str | Path,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Load CBOE VIX data from a local CSV file."""
    input_path = Path(csv_path)

    if not input_path.exists():
        raise FileNotFoundError(f"CBOE local CSV not found: {input_path}")

    try:
        raw = pd.read_csv(input_path)
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(f"Could not read CBOE CSV: {input_path}") from exc

    return standardize_cboe_vix_frame(
        raw,
        market=market,
        source=source,
        symbol=symbol,
        column_map=column_map,
    )


def download_cboe_vix_csv(
    official_url: str,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Download CBOE VIX CSV from configured URL."""
    if not official_url:
        raise DataIngestionError("CBOE official_url is empty.")

    try:
        raw = pd.read_csv(official_url)
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(
            "CBOE VIX CSV download failed. "
            f"URL attempted: {official_url}. "
            "Manual fix: download VIX_History.csv from CBOE, save it under "
            "data/manual/cboe/VIX_History.csv, and set local_csv_path in configs/data_sources.yaml."
        ) from exc

    return standardize_cboe_vix_frame(
        raw,
        market=market,
        source=source,
        symbol=symbol,
        column_map=column_map,
    )


def standardize_cboe_vix_frame(
    raw: pd.DataFrame,
    *,
    market: str,
    source: str,
    symbol: str,
    column_map: dict[str, str | list[str]] | None = None,
) -> pd.DataFrame:
    """Standardize raw CBOE VIX data to canonical OHLCV.

    CBOE VIX is used in this project as an implied-volatility close series.
    Some historical CBOE OHLC rows violate strict equity-style OHLC bounds, so
    we intentionally use close-only conversion:

    open = high = low = close = adj_close
    volume = 0
    """
    close_candidates = ["CLOSE", "Close", "close"]
    date_candidates = ["DATE", "Date", "date"]

    if column_map is not None:
        date_candidates = _as_candidate_list(column_map.get("date", date_candidates))
        close_candidates = _as_candidate_list(column_map.get("close", close_candidates))

    date_column = _first_existing_column(raw.columns, date_candidates)
    close_column = _first_existing_column(raw.columns, close_candidates)

    if date_column is None:
        raise DataIngestionError(
            f"CBOE VIX data has no recognized date column. "
            f"Tried {date_candidates}. Available columns: {list(raw.columns)}"
        )

    if close_column is None:
        raise DataIngestionError(
            f"CBOE VIX data has no recognized close column. "
            f"Tried {close_candidates}. Available columns: {list(raw.columns)}"
        )

    canonical = standardize_single_close_series(
        raw,
        date_column=date_column,
        close_column=close_column,
        market=market,
        source=source,
        symbol=symbol,
    )
    validate_ohlcv_frame(canonical)
    return canonical


def _first_existing_column(columns: pd.Index, candidates: list[str]) -> str | None:
    available = {str(col).strip(): str(col) for col in columns}
    normalized = {str(col).strip().lower(): str(col) for col in columns}

    for candidate in candidates:
        candidate_str = str(candidate).strip()

        if candidate_str in available:
            return available[candidate_str]

        candidate_norm = candidate_str.lower()
        if candidate_norm in normalized:
            return normalized[candidate_norm]

    return None


def _as_candidate_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class CboeVixLoader(BaseDataLoader):
    """Config-driven CBOE VIX loader."""

    def load(self, start: str | None = None, end: str | None = None) -> DataLoadResult:
        if self.symbol is None:
            raise DataIngestionError(f"CBOE source {self.source_id} has no symbol.")

        local_csv_path = self.source_config.get("local_csv_path")
        official_url = self.source_config.get("official_url")
        column_map = _column_map_from_config(self.source_config)

        if local_csv_path:
            frame = load_cboe_vix_from_csv(
                local_csv_path,
                market=self.market,
                source=self.source_id,
                symbol=self.symbol,
                column_map=column_map,
            )
            access_used = "local_csv"
        elif official_url:
            frame = download_cboe_vix_csv(
                str(official_url),
                market=self.market,
                source=self.source_id,
                symbol=self.symbol,
                column_map=column_map,
            )
            access_used = "official_url"
        else:
            raise DataIngestionError(
                f"CBOE source {self.source_id} has neither local_csv_path nor official_url. "
                "Manual fix: set local_csv_path to data/manual/cboe/VIX_History.csv."
            )

        frame = _filter_dates(frame, start or self.source_config.get("start_date"), end)

        return DataLoadResult(
            source_id=self.source_id,
            market=self.market,
            dataset=self.dataset,
            role=self.role,
            symbol=self.symbol,
            frame=frame,
            raw_path=self.raw_path,
            processed_path=self.processed_path,
            metadata={
                "provider": self.source_config.get("provider"),
                "access_method": self.source_config.get("access_method"),
                "access_used": access_used,
                "start": start or self.source_config.get("start_date"),
                "end": end or self.source_config.get("end_date"),
            },
        )


def _column_map_from_config(source_config: dict[str, Any]) -> dict[str, list[str]] | None:
    expected = source_config.get("expected_columns_any_of")
    if not isinstance(expected, dict):
        return None

    column_map: dict[str, list[str]] = {}
    for canonical_name, candidates in expected.items():
        if isinstance(candidates, list):
            column_map[str(canonical_name)] = [str(item) for item in candidates]
        elif isinstance(candidates, str):
            column_map[str(canonical_name)] = [candidates]

    return column_map


def _filter_dates(
    df: pd.DataFrame,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    out = df.copy()

    if start is not None:
        out = out[out["date"] >= pd.Timestamp(start)]

    if end is not None:
        out = out[out["date"] <= pd.Timestamp(end)]

    if out.empty:
        raise DataIngestionError(
            f"CBOE date filter produced no rows. start={start}, end={end}"
        )

    validate_ohlcv_frame(out)
    return out.reset_index(drop=True)