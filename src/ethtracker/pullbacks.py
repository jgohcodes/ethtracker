"""Has the pullback run its course? How often 'the low is in' signals were right, and whether waiting for
them beats adding on a schedule.

Signals, each taken on the first day it fires and entered at that day's close:
- 20% pullback: the first close 20% below the 30-day high (the trigger itself, no confirmation);
- dip done: back above the 10-day average within 20 days of a 20% dip;
- RSI turn: RSI(14) back above 30 while price is still 15%+ below its 30-day high;
- capitulation wick: during a 15%+ pullback, a day that trades 8%+ below the previous close but closes in
  the top half of its range (a spike down that is bought);
- funding flips negative: Deribit funding below zero during a 15%+ pullback (from 2019);
- confluence: two different signals within 5 days.
And one order rather than a signal: on the first close 15% below the 30-day high, a buy 12% under that
close, good for 30 days; the event is the fill, at the order's price (or the open, if it gaps through).

"The low held" means no further fall of 10% or more below the entry in the 30 days after it. Every rate is
set against the same rate on every day and on every pullback day, because ETH and BTC rose over the
history and "higher a year later" is mostly that drift, not the signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cycles, indicators, swing

PULLBACK = 0.15
WICK = 0.08
BID = 0.12
BID_DAYS = 30
CONFLUENCE_DAYS = 5
FURTHER = 0.10  # the low held: no further fall this large
HELD_DAYS = 30
HORIZONS = (30, 90, 365)
LADDER = (0.05, 0.09, 0.13, 0.20)
WEEK = 7


def in_pullback(df: pd.DataFrame, depth: float = PULLBACK) -> np.ndarray:
    return cycles.dip_armed(df, depth).to_numpy()


def capitulation_wick(df: pd.DataFrame) -> np.ndarray:
    previous = df["close"].shift(1)
    middle = (df["high"] + df["low"]) / 2
    hit = (df["low"] <= previous * (1 - WICK)) & (df["close"] >= middle)
    return swing._first(hit.to_numpy() & in_pullback(df))


def funding_negative(df: pd.DataFrame, funding8h: pd.Series | None) -> np.ndarray | None:
    if funding8h is None:
        return None
    negative = (funding8h.reindex(df.index) < 0).to_numpy()
    return swing._first(negative & in_pullback(df))


def confluence(signals: list[np.ndarray], days: int = CONFLUENCE_DAYS) -> np.ndarray:
    """Two different signals within ``days`` of each other: fires on the first day it becomes true."""
    recent = [pd.Series(s.astype(float)).rolling(days, min_periods=1).max().to_numpy() for s in signals]
    return swing._first(np.sum(recent, axis=0) >= 2)


def bid_fills(df: pd.DataFrame, depth: float = BID, days: int = BID_DAYS) -> list[tuple[int, float]]:
    """(day, price) of each fill of a buy order placed ``depth`` under the close on the first day of a pullback."""
    close, low, open_ = (df[c].to_numpy(float) for c in ("close", "low", "open"))
    starts = swing._first(in_pullback(df))
    fills, busy_until = [], -1
    for i in np.flatnonzero(starts):
        if i <= busy_until:
            continue
        bid = close[i] * (1 - depth)
        for j in range(i + 1, min(i + days, len(close) - 1) + 1):
            if low[j] <= bid:
                fills.append((j, min(bid, open_[j])))
                busy_until = j
                break
        else:
            busy_until = i + days
    return fills


def signals(df: pd.DataFrame, funding8h: pd.Series | None) -> dict[str, np.ndarray]:
    out = {
        "20% pullback (no confirmation)": cycles.first_dip_close(df),
        "Dip done (back above 10d avg)": cycles.dip_done_reclaim(df),
        "RSI back above 30": cycles.dip_done_rsi(df),
        "Capitulation wick": capitulation_wick(df),
    }
    flipped = funding_negative(df, funding8h)
    if flipped is not None:
        out["Funding flips negative"] = flipped
    out["Confluence: 2 of the above in 5 days"] = confluence([out[k] for k in list(out)[1:]])
    return out


# ---- how often the low held ----------------------------------------------------------------------

def wilson_low(hits: int, n: int, z: float = 1.96) -> float:
    """The lower end of a 95% confidence interval for a hit rate: what a small sample can honestly claim."""
    if n == 0:
        return float("nan")
    p = hits / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - spread) / (1 + z * z / n))


@dataclass(frozen=True)
class Calibration:
    name: str
    n: int  # events with 30 days ahead
    held: float  # the low held for 30 days
    held_low: float  # 95% lower bound on that
    higher: dict[int, float]  # horizon -> share higher than the entry
    n_year: int  # events with a year ahead
    median_90: float


def calibrate(name: str, df: pd.DataFrame, events: list[tuple[int, float]], *, from_fill: bool = False) -> Calibration:
    """``events`` are (day, entry price). An order filled during a day counts that day's low; a close does not."""
    close, low = df["close"].to_numpy(float), df["low"].to_numpy(float)
    n = len(close)
    held, higher, returns_90 = [], {h: [] for h in HORIZONS}, []
    for i, price in events:
        start = i if from_fill else i + 1
        if i + HELD_DAYS < n:
            held.append(low[start:i + HELD_DAYS + 1].min() > price * (1 - FURTHER))
        for h in HORIZONS:
            if i + h < n:
                higher[h].append(close[i + h] > price)
        if i + 90 < n:
            returns_90.append(close[i + 90] / price - 1)
    k = int(np.sum(held))
    return Calibration(name, len(held), k / len(held) if held else float("nan"), wilson_low(k, len(held)),
                       {h: float(np.mean(v)) if v else float("nan") for h, v in higher.items()},
                       len(higher[365]), float(np.median(returns_90)) if returns_90 else float("nan"))


def at_close(df: pd.DataFrame, mask: np.ndarray) -> list[tuple[int, float]]:
    close = df["close"].to_numpy(float)
    return [(int(i), float(close[i])) for i in np.flatnonzero(mask)]


@dataclass(frozen=True)
class Rung:
    depth: float
    days: int  # pullback days on which an order was placed
    episodes: int  # separate pullbacks those days belong to
    filled: float
    held: float  # of the fills
    higher_90: float
    higher_365: float


def ladder(df: pd.DataFrame, depths=LADDER, days: int = BID_DAYS) -> list[Rung]:
    """From every pullback day: an order ``depth`` under the close, good for ``days``. How often it filled,
    and how often the low then held and the price was higher 90 and 365 days on."""
    close, low, open_ = (df[c].to_numpy(float) for c in ("close", "low", "open"))
    n = len(close)
    pull = in_pullback(df)
    placed = [i for i in np.flatnonzero(pull) if i + days < n]
    episodes = int(swing._first(pull)[: placed[-1] + 1].sum()) if placed else 0
    out = []
    for depth in depths:
        fills = []
        for i in placed:
            bid = close[i] * (1 - depth)
            hit = np.flatnonzero(low[i + 1:i + days + 1] <= bid)
            if len(hit):
                j = i + 1 + int(hit[0])
                fills.append((j, min(bid, open_[j])))
        c = calibrate("", df, fills, from_fill=True)
        out.append(Rung(depth, len(placed), episodes, len(fills) / len(placed) if placed else float("nan"),
                        c.held, c.higher[90], c.higher[365]))
    return out


# ---- adding every week: on a schedule, or saved for the signal -------------------------------------

@dataclass(frozen=True)
class Plan:
    name: str
    signal: str | None  # key in the signals dict, "bid" for the order's fills, None for every week
    weekly_share: float  # bought on the weekly day; the rest is saved for the signal


PLANS = (
    Plan("Every week (DCA)", None, 1.0),
    Plan("Save, buy on 20% pullback", "20% pullback (no confirmation)", 0.0),
    Plan("Save, buy on dip done", "Dip done (back above 10d avg)", 0.0),
    Plan("Save, buy on capitulation wick", "Capitulation wick", 0.0),
    Plan("Save, buy on confluence", "Confluence: 2 of the above in 5 days", 0.0),
    Plan("Save, buy when the 12%-under order fills", "bid", 0.0),
    Plan("Half weekly, half on dip done", "Dip done (back above 10d avg)", 0.5),
    Plan("Half weekly, half when the order fills", "bid", 0.5),
)


@dataclass(frozen=True)
class Accumulated:
    name: str
    multiple: float  # final value / money put in
    idle: float  # share of the money still uninvested at the end
    liquidated = None  # what cycles.rolling counts; spot accumulation cannot be liquidated


def accumulate(name: str, df: pd.DataFrame, fee: float, buy_price: np.ndarray, weekly_share: float) -> Accumulated:
    """One unit of money each week. ``weekly_share`` of it buys at that close; the rest waits in cash and is all
    spent on the next day ``buy_price`` holds a price (NaN elsewhere)."""
    close = df["close"].to_numpy(float)
    coins = cash = paid = 0.0
    for i in range(len(close)):
        if i % WEEK == 0:
            paid += 1.0
            coins += weekly_share * (1 - fee) / close[i]
            cash += 1.0 - weekly_share
        if cash > 0 and not np.isnan(buy_price[i]):
            coins += cash * (1 - fee) / buy_price[i]
            cash = 0.0
    return Accumulated(name, (coins * close[-1] + cash) / paid, cash / paid)


class Plans:
    """The weekly plans, with every signal computed once on the whole history and cut to each window."""

    def __init__(self, full: pd.DataFrame, cfg, funding8h: pd.Series | None):
        self.full, self.fee = full, cfg.taker_fee_rate
        close = full["close"].to_numpy(float)
        self.prices = {}
        for key, mask in signals(full, funding8h).items():
            self.prices[key] = np.where(mask, close, np.nan)
        bid = np.full(len(full), np.nan)
        for j, price in bid_fills(full):
            bid[j] = price
        self.prices["bid"] = bid
        self.prices[None] = np.full(len(full), np.nan)

    def run(self, full, start, end, cfg, rules=None) -> list[Accumulated]:
        df = self.full[start:end]
        at = self.full.index.get_indexer(df.index)
        return [accumulate(p.name, df, self.fee, self.prices[p.signal][at], p.weekly_share) for p in PLANS]


def regular_starts(full: pd.DataFrame, years: int = cycles.HORIZON_YEARS, every: int = 28) -> list[pd.Timestamp]:
    room = full.index <= full.index[-1] - pd.DateOffset(years=years)
    return list(full.index[room][::every])


def above_trend_today(full: pd.DataFrame) -> bool:
    avg = indicators.sma(full["close"], cycles.TREND).iloc[-1]
    return bool(full["close"].iloc[-1] > avg)
