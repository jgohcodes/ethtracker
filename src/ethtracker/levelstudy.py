"""Do swing support and resistance zones hold more often than any nearby price would?

Each day the zones are rebuilt from swing highs and lows that had already been confirmed by the day before
(price had moved ``indicators.SWING_SR`` away from them), from the last two years, merged within 2%. A day
touches a support zone when its low wicks to within 0.5% of it after the previous close sat at least 1% above;
resistance is the mirror on the highs. Then:

  rejected   the same day closes back on the near side of the zone: the wick went in and came out within the day
  held 5d    no close more than 2% through the zone over the next 5 days
  bounced    the next 5 days reach 5% away from the zone on the near side before closing 2% through it

The yardstick is placebo levels: the real zones moved 4-10% up or down at random (kept clear of every real
zone), touched and judged by the same rules. If swing zones mean something, they should beat the placebo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators

TOUCH = 0.005
APPROACH = 0.01
THROUGH = 0.02
AWAY = 0.05
DAYS = 5


def confirmed_swings(df: pd.DataFrame, swing: float = indicators.SWING_SR) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """(the swing's day, the day it was confirmed, price) for every swing high and low, on daily highs and lows."""
    out, direction = [], 0
    hi_t = lo_t = df.index[0]
    hi, lo = float(df["high"].iloc[0]), float(df["low"].iloc[0])
    for t, h, l in zip(df.index, df["high"].to_numpy(float), df["low"].to_numpy(float)):
        if direction >= 0 and h > hi:
            hi, hi_t = h, t
        if direction <= 0 and l < lo:
            lo, lo_t = l, t
        if direction >= 0 and l <= hi * (1 - swing):
            out.append((hi_t, t, hi))
            direction, lo, lo_t = -1, l, t
        elif direction <= 0 and h >= lo * (1 + swing):
            out.append((lo_t, t, lo))
            direction, hi, hi_t = 1, h, t
    return [x for x in out if x[0] != df.index[0]]


@dataclass
class Result:
    touches: int = 0
    rejected: int = 0
    held: int = 0
    bounced: int = 0

    def add(self, rejected: bool, held: bool, bounced: bool) -> None:
        self.touches += 1
        self.rejected += rejected
        self.held += held
        self.bounced += bounced

    def rates(self) -> tuple[float, float, float]:
        n = max(self.touches, 1)
        return self.rejected / n, self.held / n, self.bounced / n


def _judge(df: pd.DataFrame, i: int, level: float, side: str) -> tuple[bool, bool, bool] | None:
    """Touch test and outcome for one level on day ``i``; None when the day does not touch it from the near side."""
    close, high, low = df["close"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float)
    if i + DAYS >= len(df):
        return None
    prev = close[i - 1]
    if side == "support":
        if not (prev >= level * (1 + APPROACH) and low[i] <= level * (1 + TOUCH)):
            return None
        rejected = close[i] >= level
        nxt = slice(i + 1, i + 1 + DAYS)
        broke = close[nxt] < level * (1 - THROUGH)
        held = not broke.any()
        first_break = int(np.argmax(broke)) if broke.any() else DAYS
        bounced = bool((high[i + 1:i + 1 + first_break] >= level * (1 + AWAY)).any())
    else:
        if not (prev <= level * (1 - APPROACH) and high[i] >= level * (1 - TOUCH)):
            return None
        rejected = close[i] <= level
        nxt = slice(i + 1, i + 1 + DAYS)
        broke = close[nxt] > level * (1 + THROUGH)
        held = not broke.any()
        first_break = int(np.argmax(broke)) if broke.any() else DAYS
        bounced = bool((low[i + 1:i + 1 + first_break] <= level * (1 - AWAY)).any())
    return rejected, held, bounced


def run(df: pd.DataFrame, start: pd.Timestamp | None = None, seed: int = 7) -> dict[str, Result]:
    """Touches of real and placebo zones, support and resistance, from ``start`` on."""
    rng = np.random.default_rng(seed)
    swings = confirmed_swings(df)
    out = {k: Result() for k in ("support", "resistance", "placebo support", "placebo resistance")}
    for i in range(1, len(df)):
        t = df.index[i]
        if start is not None and t < start:
            continue
        known = [p for day, conf, p in swings if conf < t and day >= t - pd.DateOffset(years=indicators.SWING_YEARS)]
        if not known:
            continue
        zones = indicators._zones(known)
        prev = float(df["close"].iloc[i - 1])
        for z in zones:
            side = "support" if z < prev else "resistance"
            r = _judge(df, i, z, side)
            if r:
                out[side].add(*r)
            shift = rng.uniform(0.04, 0.10) * rng.choice([-1, 1])
            fake = z * (1 + shift)
            if any(abs(fake / q - 1) < 0.02 for q in zones):
                continue
            fside = "support" if fake < prev else "resistance"
            r = _judge(df, i, fake, fside)
            if r:
                out[f"placebo {fside}"].add(*r)
    return out
