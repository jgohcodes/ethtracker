"""Adding cash every week to a leveraged ETH perp: put it to work at once, or save it for a 20% dip.

One account on cross margin. A deposit arrives every week. "Every week" puts it to work that day; "save for a
20% dip" leaves it in the account as margin and puts everything saved to work at the close of the first day
ETH closes 20% below its 30-day high (the same dip rule as cycle-study and pullback-study).

How much each buy opens:
  spot            coins bought outright: no funding, no liquidation (the yardstick)
  fixed leverage  the open position grows by L x the cash being put to work (1x, 2.5x, 3x on the perp)
  floor plan      the most the account can hold with its liquidation price at a floor fixed for the window,
                  a set distance under the window's first close, as the adding plan does with $1,250

Funding is paid on the open position every day (Deribit's ETH history from 2019, the config's rate before);
the taker fee on every buy. Liquidation takes the whole account, cash saved for a dip included, when equity
falls to the maintenance margin at the day's low (flash prints already removed). Deposits carry on into the
empty account afterwards. Under the floor plan a breach either ends the plan, later deposits staying in cash
(its sizing rule would otherwise buy just above the old floor at near-unlimited leverage), or, with ``reset``,
starts it again with a new floor the same distance under that day's close. Result: final equity / money put in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cycles

WEEK = 7
LEVERAGES = (1.0, 2.5, 3.0)
FLOOR_UNDER = 0.45  # generic, near the adding plan's distance today; public logs never carry the real floor


@dataclass(frozen=True)
class Run:
    multiple: float  # final equity / money put in
    liquidations: int
    idle: float  # share of the money put in still waiting at the end
    peak_leverage: float  # highest open position / equity at a close


def run(close: np.ndarray, low: np.ndarray, funding_day: np.ndarray, deploy: np.ndarray | None, *, leverage: float | None,
        floor: float | None, mmr: float, fee: float, reset_under: float | None = None) -> Run:
    """``deploy`` marks the days saved cash goes to work (None: every deposit at once). Exactly one of
    ``leverage`` and ``floor`` sizes the buys."""
    equity = qty = avg = 0.0  # equity: deposits less fees and funding, plus realised losses; excludes open profit
    waiting = paid = 0.0
    liquidations, peak, stopped = 0, 0.0, False
    for i in range(len(close)):
        if qty > 0 and equity + qty * (low[i] - avg) <= mmr * qty * low[i]:
            liquidations += 1
            equity = qty = avg = waiting = 0.0
            if floor is not None and reset_under is not None:
                floor = close[i] * (1 - reset_under)  # start again under a new floor
            else:
                stopped = floor is not None  # a breached floor ends the floor plan: later deposits stay in cash
        if qty > 0:
            equity -= qty * close[i] * funding_day[i]
        if i % WEEK == 0:
            equity += 1.0
            waiting += 1.0
            paid += 1.0
        if waiting > 0 and not stopped and (deploy is None or deploy[i]):
            price = close[i]
            if floor is None:
                buy = leverage * waiting / price
            else:
                span = price - floor * (1 - mmr)
                total = equity + qty * (price - avg)
                buy = max(0.0, (total - qty * span) / (span + fee * price)) if span > 0 else 0.0
            if buy > 0:
                equity -= fee * buy * price
                avg = (avg * qty + price * buy) / (qty + buy)
                qty += buy
            waiting = 0.0
        value = equity + qty * (close[i] - avg)
        if qty > 0 and value > 0:
            peak = max(peak, qty * close[i] / value)
    final = equity + qty * (close[-1] - avg)
    return Run(max(final, 0.0) / paid, liquidations, waiting / paid, peak)


@dataclass(frozen=True)
class Pair:
    weekly: Run
    dip: Run


class Study:
    """The dip days and daily funding, computed once on the whole history and cut to each window."""

    def __init__(self, full: pd.DataFrame, funding_day: np.ndarray, mmr: float, fee: float):
        self.full, self.funding, self.mmr, self.fee = full, funding_day, mmr, fee
        self.dips = cycles.first_dip_close(full)
        self.close = full["close"].to_numpy(float)
        self.low = full["low"].to_numpy(float)

    def window(self, start: pd.Timestamp, years: int) -> slice:
        i = self.full.index.get_loc(start)
        j = self.full.index.searchsorted(start + pd.DateOffset(years=years), side="right")
        return slice(i, j)

    def pair(self, start: pd.Timestamp, years: int, *, leverage: float | None = None, under: float | None = None,
             spot: bool = False, reset: bool = False) -> Pair:
        w = self.window(start, years)
        close, low, funding, dips = self.close[w], self.low[w], self.funding[w], self.dips[w]
        floor = None if under is None else close[0] * (1 - under)
        if spot:  # coins held outright: no funding, nothing to liquidate
            funding, leverage = np.zeros(len(close)), 1.0
        kw = dict(leverage=leverage, floor=floor, mmr=0.0 if spot else self.mmr, fee=self.fee,
                  reset_under=under if reset else None)
        return Pair(run(close, low, funding, None, **kw), run(close, low, funding, dips, **kw))


def summary(pairs: list[Pair]) -> dict:
    weekly = np.array([p.weekly.multiple for p in pairs])
    dip = np.array([p.dip.multiple for p in pairs])
    return {
        "windows": len(pairs),
        "weekly_median": float(np.median(weekly)), "dip_median": float(np.median(dip)),
        "dip_beat": float(np.mean(dip > weekly)),
        "weekly_liquidated": float(np.mean([p.weekly.liquidations > 0 for p in pairs])),
        "dip_liquidated": float(np.mean([p.dip.liquidations > 0 for p in pairs])),
        "weekly_worst": float(weekly.min()), "dip_worst": float(dip.min()),
        "weekly_above": float(np.mean(weekly > 1)), "dip_above": float(np.mean(dip > 1)),
        "dip_idle": float(np.median([p.dip.idle for p in pairs])),
        "weekly_peak_leverage": float(np.median([p.weekly.peak_leverage for p in pairs])),
        "dip_peak_leverage": float(np.median([p.dip.peak_leverage for p in pairs])),
    }
