"""Local daily-candle cache: download history once, then only the gap since the last run.

Wraps any network provider. If a refresh fails but a cache exists, the cached candles
are served with a warning, so the tracker still works offline.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .data import COLUMNS, DataError, Provider

GAP_PADDING_DAYS = 3  # re-fetch a few days back so the previous, then-partial candle is replaced


def merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union of two candle frames; where both have a day, the newer one wins."""
    both = pd.concat([old, new])
    return both[~both.index.duplicated(keep="last")].sort_index()


class CachedProvider:
    def __init__(self, inner: Provider, directory: Path):
        self.inner = inner
        self.name = inner.name
        self.path = directory / f"{inner.name}-{getattr(inner, 'product', 'ETH-USD')}-1d.csv"
        self.meta_path = self.path.with_suffix(".json")

    def spot(self) -> float:
        return self.inner.spot()

    def candles(self, days: int) -> pd.DataFrame:
        cached = self._read()
        try:
            if cached is None or self._covered_days() < days:
                fresh = self.inner.candles(days)
                merged = fresh if cached is None else merge(cached, fresh)
                covered = days
            else:
                gap = (datetime.now(timezone.utc).date() - cached.index[-1].date()).days + GAP_PADDING_DAYS
                merged = merge(cached, self.inner.candles(min(gap, days)))
                covered = self._covered_days()
        except DataError as exc:
            if cached is None:
                raise
            print(f"warning: refresh failed ({exc}); using cached candles through {cached.index[-1]:%Y-%m-%d}", file=sys.stderr)
            return cached.tail(days)
        self._write(merged, covered)
        return merged.tail(days)

    def _read(self) -> pd.DataFrame | None:
        try:
            df = pd.read_csv(self.path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df[COLUMNS].astype(float).sort_index() if len(df) else None
        except (OSError, ValueError, KeyError):
            return None

    def _covered_days(self) -> int:
        try:
            return int(json.loads(self.meta_path.read_text())["covered_days"])
        except (OSError, ValueError, KeyError):
            return 0

    def _write(self, df: pd.DataFrame, covered_days: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        df.to_csv(tmp, index_label="date")
        os.replace(tmp, self.path)
        self.meta_path.write_text(json.dumps({"covered_days": covered_days}))
