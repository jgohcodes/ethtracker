"""Buy a sharp pullback, sell only into a spike: the rule tested on ETH and on BTC.

Entries, on the first day of each dip, one position at a time:
- the first close 20% below the 30-day high;
- the first close 25% below it;
- the dip done: back above the 10-day average within 20 days of a 20% dip.
Exits, only on a spike, never on time:
- RSI(14) at 80 or more;
- a 30% run: a close 30% or more above the lowest close of the last 14 days.
There is no stop besides liquidation: one cross-margin perpetual per trade, so at 2.5x a fall of about
40% from entry ends the account. Funding is each asset's Deribit history where it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cycles, indicators
from .config import Config
from .risk import FUNDING_PERIODS_PER_DAY

SPIKE_RSI = 80.0
RUN = 0.30
RUN_DAYS = 14
LEVERAGES = (1.0, 2.5)
INSTRUMENTS = {"ETH": ("ETH-USD", "ETH-PERPETUAL"), "BTC": ("BTC-USD", "BTC-PERPETUAL")}


def first_touch(df: pd.DataFrame, depth: float) -> np.ndarray:
    armed = cycles.dip_armed(df, depth)
    return (armed & ~armed.shift(1, fill_value=False)).to_numpy()


def spike_rsi(df: pd.DataFrame) -> np.ndarray:
    return (indicators.rsi(df["close"]) >= SPIKE_RSI).to_numpy()


def spike_run(df: pd.DataFrame) -> np.ndarray:
    """A close at least 30% above the lowest close of the last 14 days (today included)."""
    close = df["close"]
    return (close >= close.rolling(RUN_DAYS, min_periods=1).min() * (1 + RUN)).to_numpy()


ENTRIES = {
    "20% dip": lambda df: first_touch(df, 0.20),
    "25% dip": lambda df: first_touch(df, 0.25),
    "Dip done": cycles.dip_done_reclaim,
}
EXITS = {
    f"RSI {SPIKE_RSI:.0f}": spike_rsi,
    f"{RUN:.0%} run in {RUN_DAYS}d": spike_run,
}


def daily_funding(full: pd.DataFrame, cfg: Config, funding8h: pd.Series | None) -> np.ndarray:
    rate = pd.Series(cfg.funding_rate_8h, index=full.index, dtype=float)
    if funding8h is not None:
        rate = funding8h.reindex(full.index).fillna(cfg.funding_rate_8h)
    return (rate * FUNDING_PERIODS_PER_DAY).to_numpy()


class Rules:
    """Every entry, exit and the funding computed once on the whole history, cut to each window."""

    def __init__(self, full: pd.DataFrame, cfg: Config, funding8h: pd.Series | None):
        self.full = full
        self.entries = {name: rule(full) for name, rule in ENTRIES.items()}
        self.exits = {name: rule(full) for name, rule in EXITS.items()}
        self.funding = daily_funding(full, cfg, funding8h)

    def run(self, full, start, end, cfg, rules=None) -> list[cycles.Result]:
        df = self.full[start:end]
        at = self.full.index.get_indexer(df.index)
        funding = self.funding[at]
        rows = [
            cycles.simulate("Hold spot, 1x", df, cfg, leverage=1.0, core=True),
            cycles.simulate("Hold perp, 2.5x", df, cfg, leverage=2.5, funding=funding),
        ]
        for entry_name, entries in self.entries.items():
            for exit_name, exits in self.exits.items():
                for lev in LEVERAGES:
                    rows.append(cycles.simulate(f"{entry_name} -> {exit_name}, {lev:g}x", df, cfg, leverage=lev,
                                                entries=entries[at], exits=exits[at], funding=funding))
        return rows

    def cost(self, start, end) -> float:
        """Average funding over a period, a year's worth, as a fraction of notional."""
        window = (self.full.index >= start) & (self.full.index <= end)
        return float(self.funding[window].mean() * 365)


@dataclass(frozen=True)
class Today:
    close: float
    below_high: float  # from the highest close so far
    below_30d_high: float
    rsi: float
    run_14d: float  # the close over the lowest close of the last 14 days, minus 1


def today(full: pd.DataFrame) -> Today:
    close = full["close"]
    return Today(float(close.iloc[-1]), float(1 - close.iloc[-1] / close.max()),
                 float(1 - close.iloc[-1] / full["high"].tail(cycles.DIP_LOOKBACK).max()),
                 float(indicators.rsi(close).iloc[-1]), float(close.iloc[-1] / close.tail(RUN_DAYS).min() - 1))


def today_depth(full: pd.DataFrame) -> float:
    """Today's distance below the highest close, rounded down to 5%, for 'from where we are now' windows."""
    return max(0.2, np.floor(today(full).below_high * 20) / 20)


# ---- trim into spikes, refill on dips, liquidation pinned to one price -------------------------------

TRIM_FRACTIONS = (1 / 3, 1 / 2)
PINNED_LEVERAGES = (2.5, 1.5)


def _first(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~np.r_[False, mask[:-1]]


def pinned_floor(price: float, leverage: float, mmr: float) -> float:
    """The liquidation price of a long opened at ``leverage``: the line every later refill is held to."""
    return price * (1 - 1 / leverage) / (1 - mmr)


def max_qty(equity: float, price: float, floor: float, mmr: float) -> float:
    """The largest long whose liquidation price is ``floor``: equity + q(floor - price) = mmr q floor."""
    room = price - floor * (1 - mmr)
    return equity / room if room > 0 and equity > 0 else 0.0


def simulate_pinned(name: str, df: pd.DataFrame, cfg: Config, funding: np.ndarray, *, leverage: float,
                    trims: np.ndarray | None = None, refills: np.ndarray | None = None,
                    trim_fraction: float = 0.0) -> cycles.Result:
    """A perpetual long opened at ``leverage`` whose liquidation price is never allowed above where it started.

    Funding and fees push a held position's liquidation price up, so each close the position is cut back to
    the size that keeps it at the floor. On a trim day ``trim_fraction`` of it is sold and the cash stays as
    margin, which lowers the liquidation price; on a refill day it is bought back up to the floor.
    """
    close, low = df["close"].to_numpy(float), df["low"].to_numpy(float)
    fee, mmr = cfg.taker_fee_rate, cfg.maintenance_margin_rate
    n = len(close)
    floor = pinned_floor(close[0], leverage, mmr)
    qty = leverage / close[0]
    equity = 1.0 - fee * leverage
    trades = 0
    curve = np.zeros(n)
    liquidated = None
    for i in range(n):
        if i:
            if qty and equity + qty * (low[i] - close[i - 1]) <= mmr * qty * low[i]:
                liquidated = df.index[i]
                break
            equity += qty * (close[i] - close[i - 1]) - funding[i] * qty * close[i]
        limit = max_qty(equity, close[i], floor, mmr)
        if qty > limit:  # funding has crept the liquidation price above the floor
            equity -= fee * (qty - limit) * close[i]
            qty = limit
        if trims is not None and trims[i] and qty:
            sold = trim_fraction * qty
            equity -= fee * sold * close[i]
            qty -= sold
            trades += 1
        elif refills is not None and refills[i]:
            limit = max_qty(equity, close[i], floor, mmr)
            if limit > qty:
                equity -= fee * (limit - qty) * close[i]
                qty = limit
                trades += 1
        curve[i] = equity
    if liquidated is not None:
        return cycles.Result(name, 0.0, 1.0, trades, 1.0, liquidated)
    final = curve[-1] - fee * qty * close[-1]
    curve[-1] = final
    return cycles.Result(name, float(final), cycles._drawdown(curve), trades, 1.0, None)


class PinnedRules:
    """Trim and refill signals plus funding, computed once on the whole history and cut to each window."""

    def __init__(self, full: pd.DataFrame, cfg: Config, funding8h: pd.Series | None):
        self.full = full
        self.trims = {name: _first(rule(full)) for name, rule in EXITS.items()}
        self.refills = first_touch(full, 0.20)
        self.funding = daily_funding(full, cfg, funding8h)

    def run(self, full, start, end, cfg, rules=None) -> list[cycles.Result]:
        df = self.full[start:end]
        at = self.full.index.get_indexer(df.index)
        funding, refills = self.funding[at], self.refills[at]
        rows = [
            cycles.simulate("Hold spot, 1x", df, cfg, leverage=1.0, core=True),
            cycles.simulate("Hold perp 2.5x, liquidation left to drift", df, cfg, leverage=2.5, funding=funding),
        ]
        for lev in PINNED_LEVERAGES:
            rows.append(simulate_pinned(f"Pinned at {lev:g}x start, never trimmed", df, cfg, funding, leverage=lev))
            for trim_name, trims in self.trims.items():
                for fraction in TRIM_FRACTIONS:
                    label = "1/3" if fraction < 0.4 else "1/2"
                    rows.append(simulate_pinned(f"Pinned {lev:g}x, trim {label} on {trim_name}, refill 20% dip", df, cfg,
                                                funding, leverage=lev, trims=trims[at], refills=refills,
                                                trim_fraction=fraction))
        return rows
