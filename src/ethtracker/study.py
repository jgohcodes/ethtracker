"""Sequential simulation of a rule-based pullback strategy on daily candles.

One position at a time. A long enters at the close of the first day price sits
``trigger`` below its ``lookback``-day high (a short mirrors this off the low), then
exits at the stop, the target, liquidation, or the time stop, whichever comes first.

Deliberately optimistic: stops and targets fill at their price with no slippage or
gap-through, and entries fill at the day's close. Deliberately conservative: when
a day's range touches both stop and target, the stop is assumed to come first.
Results are R-multiples (profit / loss-at-stop, fees and funding included), so they
do not depend on account size.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np
import pandas as pd

from . import indicators, risk
from .config import Config


@dataclass(frozen=True)
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: str
    entry: float
    exit: float
    outcome: str  # stop | target | time | liquidated
    days_held: int
    r: float


def run(
    cfg: Config,
    df: pd.DataFrame,
    side: str,
    *,
    lookback: int,
    trigger: float,
    horizon: int,
    leverage: float,
    stop_pct: float | None = None,
    stop_atr: float | None = None,
    target_pct: float | None = None,
    target_r: float | None = None,
) -> list[Trade]:
    """``trigger``, ``stop_pct`` and ``target_pct`` are fractions of price; ``stop_atr`` and ``target_r`` are multiples."""
    if (stop_pct is None) == (stop_atr is None):
        raise ValueError("give exactly one of stop_pct and stop_atr")
    if target_pct is not None and target_r is not None:
        raise ValueError("give at most one of target_pct and target_r")
    s = 1 if side == "long" else -1
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    close = df["close"].to_numpy()
    adverse = (df["low"] if s == 1 else df["high"]).to_numpy()
    favourable = (df["high"] if s == 1 else df["low"]).to_numpy()
    atr = indicators.atr(df).to_numpy()
    roll_high = df["high"].rolling(lookback).max().to_numpy()
    roll_low = df["low"].rolling(lookback).min().to_numpy()
    fee = cfg.taker_fee_rate
    last = len(df) - 1

    trades: list[Trade] = []
    i = lookback - 1
    while i < last:
        entry = close[i]
        triggered = entry <= roll_high[i] * (1 - trigger) if s == 1 else entry >= roll_low[i] * (1 + trigger)
        stop_dist = stop_pct if stop_pct is not None else stop_atr * atr[i] / entry
        if not triggered or np.isnan(stop_dist) or stop_dist >= 1:
            i += 1
            continue

        stop = entry * (1 - s * stop_dist)
        target = None
        if target_pct is not None:
            target = entry * (1 + s * target_pct)
        elif target_r is not None:
            target = entry * (1 + s * target_r * stop_dist)
        liq = risk.liquidation_price(side, entry, leverage, cfg.maintenance_margin_rate)
        # whichever protective level is nearer to entry is reached first
        level, level_outcome = (stop, "stop") if s * (stop - liq) >= 0 else (liq, "liquidated")

        end = min(i + horizon, last)
        outcome, exit_price, exit_i = "time", close[end], end
        for j in range(i + 1, end + 1):
            if s * (adverse[j] - level) <= 0:
                outcome, exit_price, exit_i = level_outcome, level, j
                break
            if target is not None and s * (favourable[j] - target) >= 0:
                outcome, exit_price, exit_i = "target", target, j
                break

        days = exit_i - i
        loss_per_unit = entry * stop_dist + fee * (entry + stop)
        if outcome == "liquidated":
            pnl = -entry / leverage
        else:
            pnl = (
                s * (exit_price - entry)
                - fee * (entry + exit_price)
                - risk.funding_cost(side, entry, cfg.funding_rate_8h, days)
            )
        trades.append(Trade(df.index[i], df.index[exit_i], side, entry, exit_price, outcome, days, pnl / loss_per_unit))
        i = exit_i + 1
    return trades


@dataclass(frozen=True)
class Summary:
    n: int
    wins: int
    avg_r: float
    median_r: float
    total_r: float
    best_r: float
    worst_r: float
    avg_days: float
    outcomes: dict[str, int]


def summarize(trades: list[Trade]) -> Summary | None:
    if not trades:
        return None
    rs = [t.r for t in trades]
    outcomes: dict[str, int] = {}
    for t in trades:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
    return Summary(
        n=len(trades), wins=sum(r > 0 for r in rs), avg_r=sum(rs) / len(rs), median_r=median(rs),
        total_r=sum(rs), best_r=max(rs), worst_r=min(rs),
        avg_days=sum(t.days_held for t in trades) / len(trades), outcomes=outcomes,
    )


def compound(trades: list[Trade], risk_pct: float) -> tuple[float, float]:
    """(total return, max drawdown) if every trade risked ``risk_pct`` of the current equity."""
    equity = peak = 1.0
    worst = 0.0
    for t in trades:
        equity = max(0.0, equity * (1 + t.r * risk_pct / 100))
        peak = max(peak, equity)
        worst = max(worst, 1 - equity / peak)
    return equity - 1, worst
