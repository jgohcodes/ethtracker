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
