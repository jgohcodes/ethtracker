"""Cycle study: does buying the end of a large dip, with leverage, beat simply holding spot?

Asked two ways. Inside each bull phase of the 4-year cycle, located with hindsight: the best case for
someone who calls the cycle right, since every phase starts at the exact bottom. And across the whole
history with only what was known at the time (price versus its 200-day average), which is the version
anyone can actually trade.

Daily candles; trades happen at the close. Leveraged positions are perpetual-futures longs in one
cross-margin account: notional = leverage x equity at entry, a fixed number of ETH until exit, funding
paid daily on the notional, taker fees on entry and exit. Liquidation is checked against each day's
low and ends the account: a liquidated strategy finishes at zero, as a cross-margin account would.
The rules below are fixed in advance and deliberately not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators
from .config import Config
from .risk import FUNDING_PERIODS_PER_DAY

DIP = 0.20  # a "large dip": a close this far below the 30-day high
DIP_LOOKBACK = 30
CONFIRM_WITHIN = 20  # days after the dip in which a recovery signal still counts
RSI_DIP = 0.15  # the RSI entry needs price at least this far below the 30-day high
EXIT_RSI = 70.0  # "sell when overbought"
MAX_HOLD = 30  # days, if overbought never comes
TREND = 200  # days in the trend filter's average

# The 4-year-cycle thesis, written down: each cycle peaks somewhere inside one of these windows. The
# peak is the highest close in its window and the trough the lowest close before the next window
# opens, so the dates come from the data. Change the windows and you change the thesis.
PEAK_WINDOWS = (
    ("2017-06-01", "2018-06-30"),
    ("2020-06-01", "2022-06-30"),
    ("2023-06-01", "2026-06-30"),
)


@dataclass(frozen=True)
class Phase:
    kind: str  # "bull" or "bear"
    start: pd.Timestamp
    end: pd.Timestamp
    start_close: float
    end_close: float
    worst_fall: float  # ETH's own deepest close-to-close fall from a high inside the phase
    from_bottom: bool  # starts at a cycle trough (False for the first phase, which starts where the data does)
    complete: bool  # False for the phase still running at the end of the data

    @property
    def change(self) -> float:
        return self.end_close / self.start_close - 1


def _drawdown(values: np.ndarray) -> float:
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        falls = np.where(peaks > 0, 1 - values / peaks, 0.0)
    return float(falls.max()) if len(falls) else 0.0


def _phase(kind: str, close: pd.Series, start, end, *, from_bottom: bool, complete: bool) -> Phase:
    part = close[start:end]
    return Phase(kind, start, end, float(close[start]), float(close[end]), _drawdown(part.to_numpy()), from_bottom, complete)


def ignore_flash_prints(df: pd.DataFrame, limit: float = 0.5) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """A perpetual is liquidated on its mark price, which is built from several venues, so one venue's
    flash-crash print does not liquidate it. A low more than ``limit`` below both the previous close and
    the day's own close is such a print; it is replaced with the lower of the day's open and close.
    Ordinary crashes (March 2020, May 2021) are nowhere near that and stay in."""
    floor = np.minimum(df["close"].shift(1), df["close"]) * (1 - limit)
    bad = (df["low"] < floor).to_numpy()
    out = df.copy()
    out.loc[bad, "low"] = np.minimum(df["open"], df["close"])[bad]
    return out, list(df.index[bad])


def find_phases(df: pd.DataFrame, peak_windows=PEAK_WINDOWS) -> list[Phase]:
    """Alternating bull and bear phases, with hindsight. Stops at the first window the data does not reach."""
    close = df["close"]
    tz = close.index.tz
    first, last = close.index[0], close.index[-1]
    bounds = [(pd.Timestamp(lo, tz=tz), pd.Timestamp(hi, tz=tz)) for lo, hi in peak_windows]
    phases: list[Phase] = []
    start, from_bottom = first, False
    for k, (lo, hi) in enumerate(bounds):
        if lo > last:
            break
        if hi < first:  # this cycle peaked before the data begins, so the data opens in its bear phase
            peak = first
        elif last <= hi:  # the data ends inside this cycle's peak window, so its top cannot be known yet
            break
        else:
            peak = close[(close.index >= max(lo, start)) & (close.index <= hi)].idxmax()
            if peak > start:
                phases.append(_phase("bull", close, start, peak, from_bottom=from_bottom, complete=True))
        next_open = bounds[k + 1][0] if k + 1 < len(bounds) else None
        if next_open is None or next_open > last:  # the bottom cannot be known yet
            phases.append(_phase("bear", close, peak, last, from_bottom=False, complete=False))
            return phases
        trough = close[(close.index >= peak) & (close.index < next_open)].idxmin()
        phases.append(_phase("bear", close, peak, trough, from_bottom=False, complete=True))
        start, from_bottom = trough, True
    phases.append(_phase("bull", close, start, last, from_bottom=from_bottom, complete=False))
    return phases


# ---- entry and exit rules, each using only data up to that day's close ----------------------------

def dip_armed(df: pd.DataFrame, dip: float = DIP) -> pd.Series:
    return df["close"] <= df["high"].rolling(DIP_LOOKBACK).max() * (1 - dip)


def first_dip_close(df: pd.DataFrame) -> np.ndarray:
    """The first close 20% below the 30-day high: one entry per dip, the rule tested before."""
    armed = dip_armed(df)
    return (armed & ~armed.shift(1, fill_value=False)).to_numpy()


def dip_done_reclaim(df: pd.DataFrame) -> np.ndarray:
    """The dip is done: the first close back above the 10-day average within 20 days of a 20% dip.

    One entry per dip. If the recovery does not come within 20 days, that dip is skipped.
    """
    armed = dip_armed(df)
    start = (armed & ~armed.shift(1, fill_value=False)).to_numpy()
    close, avg = df["close"], indicators.sma(df["close"], 10)
    cross = ((close > avg) & (close.shift(1) <= avg.shift(1))).to_numpy()
    out = np.zeros(len(df), dtype=bool)
    last, used = None, True
    for i in range(len(df)):
        if start[i]:
            last, used = i, False
        elif last is not None and not used and i - last <= CONFIRM_WITHIN and cross[i]:
            out[i], used = True, True
    return out


def dip_done_rsi(df: pd.DataFrame) -> np.ndarray:
    """The dip is done: RSI(14) climbs back through 30 while price is still 15%+ below its 30-day high."""
    rsi = indicators.rsi(df["close"])
    return ((rsi >= 30) & (rsi.shift(1) < 30) & dip_armed(df, RSI_DIP)).to_numpy()


def above_trend(df: pd.DataFrame) -> np.ndarray:
    return (df["close"] > indicators.sma(df["close"], TREND)).to_numpy()


# ---- the account ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    name: str
    multiple: float  # final equity / starting equity
    max_drawdown: float  # deepest fall of equity marked at each close
    trades: int
    exposure: float  # share of days with any position
    liquidated: pd.Timestamp | None  # the day the whole account was liquidated (cross margin)
    lost_positions: int = 0  # isolated positions liquidated, each losing only its own margin


def simulate(name: str, df: pd.DataFrame, cfg: Config, *, leverage: float, entries: np.ndarray | None = None,
             exits: np.ndarray | None = None, max_hold: int | None = None, core: bool = False,
             isolated: float | None = None, funding: np.ndarray | None = None) -> Result:
    """One account, starting at 1.0, over ``df``.

    ``core`` buys spot with all the equity on the first day and keeps it. A perpetual long opens at the
    close of any entry day while none is open, sized so total exposure is ``leverage`` x equity; it
    closes at the close of an exit day or after ``max_hold`` days. With no ``entries`` and no ``core``,
    the account opens one perpetual on the first day and holds it to the end.

    ``isolated`` instead posts that fraction of equity as isolated margin at ``leverage`` (exchange
    leverage): a liquidation then loses that margin and the account carries on.

    ``funding`` gives each day's funding as a fraction of notional (paid by longs); without it the
    configured 8-hour rate is charged every day.
    """
    close, low = df["close"].to_numpy(float), df["low"].to_numpy(float)
    n = len(close)
    fee, mmr = cfg.taker_fee_rate, cfg.maintenance_margin_rate
    if funding is None:
        funding = np.full(n, cfg.funding_rate_8h * FUNDING_PERIODS_PER_DAY)
    if entries is None:
        entries = np.zeros(n, dtype=bool)
        entries[0] = not core
    exits = np.zeros(n, dtype=bool) if exits is None else exits

    spot = (1 - fee) / close[0] if core else 0.0
    cash = 0.0 if core else 1.0
    qty = entry = margin = 0.0
    opened = 0
    trades = days_in = lost = 0
    curve = np.zeros(n)
    liquidated = None
    for i in range(n):
        if qty and isolated is not None and margin + qty * (low[i] - entry) <= mmr * qty * low[i]:
            cash -= margin  # the position's margin is gone; the rest of the account is untouched
            qty = 0.0
            lost += 1
        # cross margin: the whole account, spot included, has to cover the maintenance margin; spot held
        # against borrowed cash (fees, funding and losses already paid) is liquidated once it is worth less
        if (qty or cash < 0) and spot * low[i] + cash + qty * (low[i] - entry) <= mmr * qty * low[i]:
            liquidated = df.index[i]
            break
        if qty:
            cash -= funding[i] * qty * close[i]
            if exits[i] or (max_hold is not None and i - opened >= max_hold):
                cash += qty * (close[i] - entry) - fee * qty * close[i]
                qty = 0.0
        equity = spot * close[i] + cash + qty * (close[i] - entry)
        if not qty and entries[i] and equity > 0:
            margin = isolated * equity if isolated is not None else 0.0
            notional = leverage * margin if isolated is not None else leverage * equity - spot * close[i]
            if notional > 0:
                qty, entry, opened = notional / close[i], close[i], i
                cash -= fee * notional
                equity -= fee * notional
                trades += 1
        curve[i] = equity
        days_in += bool(qty or spot)
    if liquidated is None and qty:
        curve[-1] -= fee * qty * close[-1]
    multiple = 0.0 if liquidated is not None else float(curve[-1])
    drawdown = 1.0 if liquidated is not None else _drawdown(curve)
    return Result(name, multiple, drawdown, trades, days_in / n, liquidated, lost)


# ---- the strategies, fixed in advance -----------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    """Every rule computed once on the whole history (the averages need it), then cut to each window."""
    index: pd.DatetimeIndex
    first_dip: np.ndarray
    reclaim: np.ndarray
    rsi_turn: np.ndarray
    up: np.ndarray
    overbought: np.ndarray

    @classmethod
    def of(cls, full: pd.DataFrame) -> "Rules":
        return cls(full.index, first_dip_close(full), dip_done_reclaim(full), dip_done_rsi(full), above_trend(full),
                   (indicators.rsi(full["close"]) >= EXIT_RSI).to_numpy())

    def cut(self, name: str, df: pd.DataFrame) -> np.ndarray:
        return getattr(self, name)[self.index.get_indexer(df.index)]


@dataclass(frozen=True)
class Strategy:
    name: str
    run: Callable[[Rules, pd.DataFrame, Config, bool], Result]
    dip: bool  # a dip-buying rule, which the trend filter applies to


def _hold(name, leverage, core=False):
    return Strategy(name, lambda rules, df, cfg, gated: simulate(name, df, cfg, leverage=leverage, core=core), dip=False)


def _trend(name, leverage):
    def run(rules, df, cfg, gated):
        up = rules.cut("up", df)
        return simulate(name, df, cfg, leverage=leverage, entries=up, exits=~up)
    return Strategy(name, run, dip=False)


def _dip(name, rule, leverage, core=False, isolated=None):
    def run(rules, df, cfg, gated):
        entries = rules.cut(rule, df)
        if gated:
            entries = entries & rules.cut("up", df)
        return simulate(name, df, cfg, leverage=leverage, entries=entries, exits=rules.cut("overbought", df),
                        max_hold=MAX_HOLD, core=core, isolated=isolated)
    return Strategy(name, run, dip=True)


STRATEGIES = (
    _hold("Hold spot, 1x", 1.0, core=True),
    _hold("Hold perp, 1.5x", 1.5),
    _hold("Hold perp, 2.5x", 2.5),
    _trend("Trend: long while above 200d avg, 1x", 1.0),
    _trend("Trend: long while above 200d avg, 2.5x", 2.5),
    _dip("Dip: first close 20% down, 2.5x", "first_dip", 2.5),
    _dip("Dip done: back above 10d avg, 1x", "reclaim", 1.0),
    _dip("Dip done: back above 10d avg, 2.5x", "reclaim", 2.5),
    _dip("Dip done: back above 10d avg, 5x", "reclaim", 5.0),
    _dip("Dip done: 10x isolated on 25% of equity", "reclaim", 10.0, isolated=0.25),
    _dip("Dip done: RSI back above 30, 2.5x", "rsi_turn", 2.5),
    _dip("Spot core + dip-done add-on to 2.5x", "reclaim", 2.5, core=True),
)


def run_window(full: pd.DataFrame, start, end, cfg: Config, *, gated: bool = False,
               rules: Rules | None = None) -> list[Result]:
    """Every strategy over one window. ``gated`` keeps only the dip rules, entering above the 200d average."""
    rules = rules or Rules.of(full)
    df = full[start:end]
    chosen = [s for s in STRATEGIES if s.dip or s is STRATEGIES[0]] if gated else STRATEGIES
    return [s.run(rules, df, cfg, gated) for s in chosen]


def chain(windows: list[list[Result]]) -> list[Result]:
    """Back to back: in at each phase's start, out at its end, compounding. A liquidation anywhere ends it."""
    out = []
    for rows in zip(*windows):
        liquidated = next((r.liquidated for r in rows if r.liquidated is not None), None)
        multiple = 0.0 if liquidated is not None else float(np.prod([r.multiple for r in rows]))
        out.append(Result(rows[0].name, multiple, max(r.max_drawdown for r in rows), sum(r.trades for r in rows),
                          float(np.mean([r.exposure for r in rows])), liquidated, sum(r.lost_positions for r in rows)))
    return out


# ---- from where we are now: deep below the high, four years forward ------------------------------

DEEP = 0.50  # "near the end of a down cycle", measured without hindsight: this far below the all-time-high close
HORIZON_YEARS = 4


@dataclass(frozen=True)
class Rolling:
    name: str
    starts: int
    median_multiple: float
    beat_spot: float  # share of starts where the rule finished ahead of holding spot
    liquidated: float  # share of starts where the account was liquidated
    worst: float
    median_vs_spot_by_episode: tuple[float, ...]


def deep_starts(full: pd.DataFrame, depth: float = DEEP, years: int = HORIZON_YEARS, every: int = 7) -> list[pd.Timestamp]:
    """Every ``every``-th day on which the close was ``depth`` or more below the highest close so far,
    with a full ``years`` of data after it."""
    close = full["close"]
    deep = (close <= close.cummax() * (1 - depth)).to_numpy()
    room = full.index <= full.index[-1] - pd.DateOffset(years=years)
    return list(full.index[deep & room][::every])


def episodes(starts: list[pd.Timestamp], gap_days: int = 60) -> list[list[pd.Timestamp]]:
    """Group start dates into separate drawdowns: a gap of more than ``gap_days`` begins a new one."""
    groups: list[list[pd.Timestamp]] = []
    for day in starts:
        if groups and (day - groups[-1][-1]).days <= gap_days:
            groups[-1].append(day)
        else:
            groups.append([day])
    return groups


def _median(values: list[float]) -> float:
    """Median ignoring NaN (a benchmark that was liquidated leaves no ratio); NaN when nothing is left."""
    kept = [v for v in values if not np.isnan(v)]
    return float(np.median(kept)) if kept else float("nan")


def rolling(full: pd.DataFrame, cfg: Config, starts: list[pd.Timestamp], years: int = HORIZON_YEARS,
            runner=None) -> list[Rolling]:
    """Every strategy from every start date, ``years`` forward, summarised across the starts.

    ``runner(full, start, end, cfg, rules)`` returns one result per strategy, holding spot first; it
    defaults to this module's strategies. Each result needs ``name``, ``multiple`` and ``liquidated``.
    """
    runner = runner or (lambda full, start, end, cfg, rules: run_window(full, start, end, cfg, rules=rules))
    rules = Rules.of(full)
    groups = episodes(starts)
    episode_of = {day: k for k, group in enumerate(groups) for day in group}
    runs = {day: runner(full, day, day + pd.DateOffset(years=years), cfg, rules) for day in starts}
    spot = np.array([runs[day][0].multiple for day in starts])
    out = []
    for j, first in enumerate(runs[starts[0]]):
        results = [runs[day][j] for day in starts]
        multiples = np.array([r.multiple for r in results])
        vs = np.divide(multiples, spot, out=np.full(len(spot), np.nan), where=spot > 0)
        by_episode = tuple(_median([v for day, v in zip(starts, vs) if episode_of[day] == k]) for k in range(len(groups)))
        out.append(Rolling(first.name, len(results), float(np.median(multiples)), float(np.mean(vs > 1)),
                           float(np.mean([r.liquidated is not None for r in results])), float(multiples.min()), by_episode))
    return out
