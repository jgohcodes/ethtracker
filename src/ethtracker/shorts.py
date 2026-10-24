"""Shorting an overheated market: does a short-term short pay when ETH looks overheated?

Three overheating signals, fixed in advance and taken on the first day of each episode:
- the dashboard's own warning: two or more of its technical flags (stretched, RSI hot, volatility spike);
- RSI(14) at 80 or above;
- funding at 0.05% per 8h or more on Deribit's ether perpetual (Deribit's public funding history).

A short opens at that day's close and is covered at the first close back below the 10-day average
(the heat has gone) or after 14 days. Shorts receive funding when it is positive and pay it when it is
negative, using Deribit's daily funding where it exists and the configured rate elsewhere. Liquidation
is checked against each day's high. Each trade posts a fresh stake as isolated margin, so per-trade
results are independent; "re-staked" compounds the whole sleeve trade after trade, as a 10-15x sleeve
would, and ends at zero at the first liquidation.

The overlay asks the question for the 2.5x core: in one account a short on the same perpetual simply
shrinks the long, so "short when overheated" there means cutting the core to 1x on the signal and
buying the same quantity back when it cools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cycles, indicators, risk, signals
from .config import Config
from .risk import FUNDING_PERIODS_PER_DAY

HORIZON = 14  # days: the longest a short-term short is held
RSI_HOT = 80.0
FUNDING_HOT = 0.0005  # 0.05% per 8h
COOL_AVG = 10  # covered at the first close back below this average
LEVERAGES = (1.0, 3.0, 10.0, 15.0)
CORE_LEVERAGE = 2.5
DERISK_TO = 1.0


def _first(mask: pd.Series) -> np.ndarray:
    mask = mask.fillna(False).astype(bool)
    return (mask & ~mask.shift(1, fill_value=False)).to_numpy()


def overheated(df: pd.DataFrame, cfg: Config, funding8h: pd.Series | None) -> dict[str, np.ndarray]:
    """First day of each overheating episode, per signal. The funding signal needs Deribit's history."""
    tech = signals.technical_series(cfg, df)
    out = {
        "Dashboard: 2+ overheating flags": _first(tech["risk_count"] >= 2),
        f"RSI(14) at {RSI_HOT:.0f}+": _first(indicators.rsi(df["close"]) >= RSI_HOT),
    }
    if funding8h is not None:
        out[f"Funding at {FUNDING_HOT:.2%}/8h+"] = _first(funding8h.reindex(df.index) >= FUNDING_HOT)
    return out


def cooled(df: pd.DataFrame) -> np.ndarray:
    return (df["close"] < indicators.sma(df["close"], COOL_AVG)).to_numpy()


def daily_funding(df: pd.DataFrame, cfg: Config, funding8h: pd.Series | None) -> np.ndarray:
    """Funding per day as a fraction of notional, paid by longs to shorts (negative: shorts pay)."""
    rate = pd.Series(cfg.funding_rate_8h, index=df.index, dtype=float)
    if funding8h is not None:
        rate = funding8h.reindex(df.index).fillna(cfg.funding_rate_8h)
    return (rate * FUNDING_PERIODS_PER_DAY).to_numpy()


# ---- what followed each signal -------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    name: str
    count: int
    median_move: float  # ETH over the next 14 days
    fell_10: float  # share where ETH traded 10%+ below the signal's close within 14 days
    squeeze: dict[float, float]  # leverage -> share where it traded high enough to liquidate a short at it


def events(name: str, df: pd.DataFrame, cfg: Config, entries: np.ndarray, horizon: int = HORIZON,
           squeeze_at=(10.0, 15.0)) -> Event:
    close, low, high = (df[c].to_numpy(float) for c in ("close", "low", "high"))
    moves, fell, squeezed = [], [], {lev: [] for lev in squeeze_at}
    for i in np.flatnonzero(entries):
        if i + horizon >= len(close):
            continue
        ahead = slice(i + 1, i + horizon + 1)
        moves.append(close[i + horizon] / close[i] - 1)
        fell.append(low[ahead].min() <= close[i] * 0.9)
        for lev in squeeze_at:
            squeezed[lev].append(high[ahead].max() >= risk.liquidation_price("short", close[i], lev, cfg.maintenance_margin_rate))
    if not moves:
        return Event(name, 0, float("nan"), float("nan"), {lev: float("nan") for lev in squeeze_at})
    return Event(name, len(moves), float(np.median(moves)), float(np.mean(fell)),
                 {lev: float(np.mean(v)) for lev, v in squeezed.items()})


# ---- short trades --------------------------------------------------------------------------------

def short_returns(df: pd.DataFrame, cfg: Config, entries: np.ndarray, cool: np.ndarray, funding: np.ndarray,
                  leverage: float, horizon: int = HORIZON) -> list[float]:
    """Return on margin of each short, one at a time (signals while a short is open are skipped)."""
    close, high = df["close"].to_numpy(float), df["high"].to_numpy(float)
    fee, mmr = cfg.taker_fee_rate, cfg.maintenance_margin_rate
    out: list[float] = []
    i, n = 0, len(close)
    while i < n:
        if not entries[i]:
            i += 1
            continue
        entry, qty = close[i], leverage / close[i]
        pnl = -fee * leverage  # entry fee on a margin of 1
        result = None
        for j in range(i + 1, n):
            if 1.0 + pnl + qty * (entry - high[j]) <= mmr * qty * high[j]:
                result = -1.0
                break
            pnl += funding[j] * qty * close[j]
            if cool[j] or j - i >= horizon:
                result = pnl + qty * (entry - close[j]) - fee * qty * close[j]
                break
        if result is None:  # still open when the data ends
            break
        out.append(max(result, -1.0))
        i = j + 1
    return out


@dataclass(frozen=True)
class ShortResult:
    name: str
    leverage: float
    trades: int
    win_rate: float
    liquidated: float
    mean_return: float  # on margin, per trade
    median_return: float
    restaked: float  # the whole sleeve re-staked every trade, starting at 1

    @classmethod
    def of(cls, name: str, leverage: float, returns: list[float]) -> "ShortResult":
        r = np.array(returns, dtype=float)
        if not len(r):
            return cls(name, leverage, 0, float("nan"), float("nan"), float("nan"), float("nan"), 1.0)
        return cls(name, leverage, len(r), float(np.mean(r > 0)), float(np.mean(r <= -1.0)), float(r.mean()),
                   float(np.median(r)), float(np.prod(1 + r)))


# ---- the overlay on the core ---------------------------------------------------------------------

def core_with_derisk(name: str, df: pd.DataFrame, cfg: Config, funding: np.ndarray, *, entries: np.ndarray | None = None,
                     cool: np.ndarray | None = None, leverage: float = CORE_LEVERAGE, derisk_to: float = DERISK_TO,
                     horizon: int = HORIZON) -> cycles.Result:
    """A perp opened at ``leverage`` on day one and held; on an entry day cut to ``derisk_to`` x equity, and on
    the first cool close (or after ``horizon`` days) buy back to the quantity held before."""
    close, low = df["close"].to_numpy(float), df["low"].to_numpy(float)
    fee, mmr = cfg.taker_fee_rate, cfg.maintenance_margin_rate
    n = len(close)
    qty = leverage / close[0]
    equity = 1.0 - fee * leverage
    held_before, since, cuts = None, 0, 0
    curve = np.zeros(n)
    liquidated = None
    for i in range(n):
        if i:
            if equity + qty * (low[i] - close[i - 1]) <= mmr * qty * low[i]:
                liquidated = df.index[i]
                break
            equity += qty * (close[i] - close[i - 1]) - funding[i] * qty * close[i]
            if held_before is not None and (cool[i] or i - since >= horizon):
                equity -= fee * abs(held_before - qty) * close[i]
                qty, held_before = held_before, None
        if entries is not None and held_before is None and entries[i] and equity > 0:
            target = derisk_to * equity / close[i]
            if target < qty:
                equity -= fee * (qty - target) * close[i]
                held_before, qty, since = qty, target, i
                cuts += 1
        curve[i] = equity
    if liquidated is not None:
        return cycles.Result(name, 0.0, 1.0, cuts, 1.0, liquidated)
    final = curve[-1] - fee * qty * close[-1]
    curve[-1] = final
    return cycles.Result(name, float(final), cycles._drawdown(curve), cuts, 1.0, None)


class Overlay:
    """Every signal's overlay plus the core alone (first), with rules computed once on the whole history."""

    def __init__(self, full: pd.DataFrame, cfg: Config, funding8h: pd.Series | None):
        self.full, self.cfg = full, cfg
        self.entries = overheated(full, cfg, funding8h)
        self.cool = cooled(full)
        self.funding = daily_funding(full, cfg, funding8h)

    def run(self, full, start, end, cfg, rules=None) -> list[cycles.Result]:
        df = self.full[start:end]
        at = self.full.index.get_indexer(df.index)
        funding = self.funding[at]
        rows = [core_with_derisk(f"Core {CORE_LEVERAGE:g}x held, never cut", df, cfg, funding)]
        for name, entries in self.entries.items():
            rows.append(core_with_derisk(f"Cut to {DERISK_TO:g}x on: {name}", df, cfg, funding,
                                         entries=entries[at], cool=self.cool[at]))
        return rows
