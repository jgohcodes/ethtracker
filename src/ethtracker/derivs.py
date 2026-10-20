"""Derivatives-market data: open interest, funding and liquidations.

Free sources only expose a *snapshot* of open interest and funding (Deribit's public
ticker), so ``record`` builds the history by polling and appending to a CSV. Liquidation
volumes have no complete free feed (Binance's stream pushes at most one liquidation per
symbol per second), so they are accepted from a CSV you supply and otherwise absent.

All columns are optional: signals are computed from whichever are present.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import DataError, _get_json

COLUMNS = ["open_interest", "funding_8h", "current_funding", "mark_price", "index_price", "long_liq_usd", "short_liq_usd"]
SNAPSHOT_COLUMNS = ["time", "open_interest", "funding_8h", "current_funding", "mark_price", "index_price"]
TICKER_URL = "https://www.deribit.com/api/v2/public/ticker"


def parse_deribit_ticker(payload: dict, now: datetime | None = None) -> dict:
    """One snapshot row from Deribit's ``public/ticker``. Needs ``open_interest`` and ``funding_8h``.

    Open interest is in the instrument's own units (USD for Deribit's inverse perpetuals), so
    only its relative change and percentile are meaningful here, never its absolute size.
    """
    try:
        result = payload["result"]
        row = {"open_interest": float(result["open_interest"]), "funding_8h": float(result["funding_8h"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"unexpected Deribit ticker payload: {str(payload)[:300]}") from exc
    for key in ("current_funding", "mark_price", "index_price"):
        value = result.get(key)
        row[key] = float(value) if isinstance(value, (int, float)) else float("nan")
    stamp = result.get("timestamp")
    when = pd.Timestamp(stamp, unit="ms", tz="UTC") if isinstance(stamp, (int, float)) else pd.Timestamp(now or datetime.now(timezone.utc))
    return {"time": when.isoformat(), **row}


def fetch_deribit_snapshot(instrument: str) -> dict:
    return parse_deribit_ticker(_get_json(TICKER_URL, {"instrument_name": instrument}))


def recorded_path(directory: Path, instrument: str) -> Path:
    return directory / f"deribit-{instrument}.csv"


def append_snapshot(path: Path, row: dict) -> bool:
    """Append one snapshot; False when a row with the same timestamp is already the last one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    if not fresh:
        with path.open() as f:
            last = None
            for last in csv.reader(f):
                pass
        if last and last[0] == row["time"]:
            return False
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if fresh:
            writer.writerow(SNAPSHOT_COLUMNS)
        writer.writerow([row[c] for c in SNAPSHOT_COLUMNS])
    return True


def load_csv(path: Path) -> pd.DataFrame:
    """Derivatives history from a CSV with time/date/timestamp plus any of ``COLUMNS``."""
    try:
        df = pd.read_csv(path)
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    df.columns = [c.strip().lower() for c in df.columns]
    stamp = next((c for c in ("time", "date", "timestamp") if c in df.columns), None)
    present = [c for c in COLUMNS if c in df.columns]
    if stamp is None or not present:
        raise DataError(f"{path} needs time/date plus at least one of: {', '.join(COLUMNS)}")
    numeric = pd.api.types.is_numeric_dtype(df[stamp])
    df.index = pd.to_datetime(df.pop(stamp), unit="s" if numeric else None, utc=True)
    return df[present].apply(pd.to_numeric, errors="coerce").sort_index()
