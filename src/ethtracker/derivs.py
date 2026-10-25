"""Derivatives-market data: open interest, funding and liquidations.

Free sources only expose a *snapshot* of open interest and funding (Deribit's public
ticker), so ``record`` builds the history by polling and appending to a CSV. Liquidation
volumes have no complete free feed (Binance's stream pushes at most one liquidation per
symbol per second), so they are accepted from a CSV you supply and otherwise absent.

All columns are optional: signals are computed from whichever are present.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import DataError, _get_json

COLUMNS = ["open_interest", "funding_8h", "current_funding", "mark_price", "index_price", "long_liq_usd", "short_liq_usd"]
SNAPSHOT_COLUMNS = ["time", "open_interest", "funding_8h", "current_funding", "mark_price", "index_price"]
TICKER_URL = "https://www.deribit.com/api/v2/public/ticker"
FUNDING_HISTORY_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
HISTORY_INSTRUMENT = "ETH-PERPETUAL"  # Deribit's ether perpetual with the longest funding history


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


def parse_deribit_funding_history(payload: dict) -> list[tuple[int, float]]:
    """(timestamp in ms, 8-hour funding rate as a fraction) pairs from ``public/get_funding_rate_history``."""
    try:
        return [(int(row["timestamp"]), float(row["interest_8h"])) for row in payload["result"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"unexpected Deribit funding history payload: {str(payload)[:300]}") from exc


def fetch_funding_history(start, end, instrument: str = HISTORY_INSTRUMENT, step_days: int = 30,
                          pause: float = 0.1) -> tuple[pd.Series, int]:
    """Daily mean of Deribit's 8-hour funding rate (0.0001 is 0.01%), fetched a month at a time.

    Returns the series and how many monthly pages failed; raises DataError only if nothing came back.
    """
    rows: list[tuple[int, float]] = []
    failed, first_error = 0, None
    t, end = pd.Timestamp(start), pd.Timestamp(end)
    while t < end:
        stop = min(t + pd.Timedelta(days=step_days), end)
        try:
            rows += parse_deribit_funding_history(_get_json(FUNDING_HISTORY_URL, {
                "instrument_name": instrument,
                "start_timestamp": int(t.timestamp() * 1000),
                "end_timestamp": int(stop.timestamp() * 1000),
            }))
        except DataError as exc:
            failed, first_error = failed + 1, first_error or exc
        t = stop
        time.sleep(pause)
    if not rows:
        raise DataError(f"Deribit returned no funding history ({first_error or 'empty'})")
    series = pd.Series(dict(rows)).sort_index()
    series.index = pd.to_datetime(series.index, unit="ms", utc=True)
    return series.groupby(series.index.normalize()).mean(), failed


def fetch_deribit_snapshot(instrument: str) -> dict:
    return parse_deribit_ticker(_get_json(TICKER_URL, {"instrument_name": instrument}))


def recorded_path(directory: Path, instrument: str) -> Path:
    return directory / f"deribit-{instrument}.csv"


RECENT_HOURS = 72  # kept at full resolution: the 24-hour open-interest change needs rows a day apart


def compact(text: str, now: datetime | None = None, recent_hours: int = RECENT_HOURS) -> str:
    """The recorded history with every row of the last ``recent_hours`` and only the last row of each UTC day
    before that, so hourly runs keep years of daily history within the stored size."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return text
    cutoff = pd.Timestamp(now or datetime.now(timezone.utc)) - pd.Timedelta(hours=recent_hours)
    rows = []
    for line in lines[1:]:
        try:
            t = pd.Timestamp(line.split(",", 1)[0])
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        except ValueError:
            t = None
        rows.append((t, line))
    last_of_day = {}
    for i, (t, _) in enumerate(rows):
        if t is not None and t < cutoff:
            last_of_day[t.date()] = i
    keep = [line for i, (t, line) in enumerate(rows) if t is None or t >= cutoff or last_of_day.get(t.date()) == i]
    return "\n".join([lines[0], *keep]) + "\n"


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
