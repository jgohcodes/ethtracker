"""Two sleeves: a long-only core holding most of the money, and a small high-leverage swing sleeve
whose profits are swept into the core.

The core and the sleeve are separate margin pools, as two sub-accounts (or an isolated swing position)
would be: a liquidated swing trade loses the sleeve's margin and never touches the core, and a
liquidated core leaves the sleeve trading. Each swing trade posts the whole sleeve as margin; when a
trade closes with the sleeve above its starting stake, the excess is swept into the core. A sleeve
that has been wiped out is not refilled.

Marked to market at each close; liquidation is checked against each day's low (after
cycles.ignore_flash_prints); funding is paid daily on every open notional; taker fees are paid on every
trade, sweep and resize. The swing sleeve uses the dip-done rule from cycles: in at the first close back
above the 10-day average after a 20% dip, out at RSI 70 or after 30 days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import cycles
from .config import Config
from .risk import FUNDING_PERIODS_PER_DAY

BAND = 0.2  # a "kept" core is resized at the close once its leverage drifts 20% from target (2x-3x for 2.5x)


@dataclass(frozen=True)
class Combo:
    name: str
    core_share: float  # of starting equity; the rest is the swing sleeve
    core_leverage: float  # 1.0 for a spot core
    core_mode: str  # "spot"; "held": one perpetual opened on day one; "kept": resized back to its leverage
    swing_leverage: float | None = None  # isolated leverage of each swing trade; None for no swing sleeve
    sweep: str = "buy"  # swept profit buys more core at the core's leverage ("buy") or only adds margin ("margin")


@dataclass(frozen=True)
class ComboResult:
    name: str
    multiple: float  # final equity of both sleeves / starting equity
    max_drawdown: float
    core_liquidated: pd.Timestamp | None
    swing_trades: int
    swing_liquidated: int
    swept: float  # profit moved from the sleeve into the core, as a fraction of starting equity

    @property
    def liquidated(self) -> pd.Timestamp | None:  # what cycles.rolling counts
        return self.core_liquidated


def simulate(combo: Combo, df: pd.DataFrame, cfg: Config, entries: np.ndarray, exits: np.ndarray,
             max_hold: int = cycles.MAX_HOLD) -> ComboResult:
    close, low = df["close"].to_numpy(float), df["low"].to_numpy(float)
    n = len(close)
    fee, mmr = cfg.taker_fee_rate, cfg.maintenance_margin_rate
    funding = cfg.funding_rate_8h * FUNDING_PERIODS_PER_DAY
    perp = combo.core_mode != "spot"
    lev = combo.core_leverage if perp else 1.0

    if perp:
        qty = lev * combo.core_share / close[0]
        core = combo.core_share - fee * qty * close[0]
    else:
        qty = combo.core_share * (1 - fee) / close[0]
        core = qty * close[0]
    core_dead = None

    base = 1.0 - combo.core_share
    cash = base  # the sleeve's equity while it is flat
    s_qty = s_entry = margin = 0.0
    opened = trades = lost = 0
    swept = 0.0
    curve = np.empty(n)
    for i in range(n):
        if i:
            if qty:
                if perp and core + qty * (low[i] - close[i - 1]) <= mmr * qty * low[i]:
                    core_dead, core, qty = df.index[i], 0.0, 0.0
                else:
                    core += qty * (close[i] - close[i - 1]) - (funding * qty * close[i] if perp else 0.0)
            if s_qty:
                if margin + s_qty * (low[i] - s_entry) <= mmr * s_qty * low[i]:
                    cash -= margin  # the sleeve's margin is gone; the core is untouched
                    s_qty = 0.0
                    lost += 1
                else:
                    cost = funding * s_qty * close[i]
                    cash -= cost
                    margin -= cost
                    if exits[i] or i - opened >= max_hold:
                        cash += s_qty * (close[i] - s_entry) - fee * s_qty * close[i]
                        s_qty = 0.0
                        if cash > base and core_dead is None:
                            profit, cash = cash - base, base
                            swept += profit
                            core += profit
                            if combo.sweep == "buy":
                                added = lev * profit / close[i]
                                qty += added
                                core -= fee * added * close[i]
            if combo.core_mode == "kept" and qty and core > 0 and abs(qty * close[i] / core / lev - 1) > BAND:
                target = lev * core / close[i]
                core -= fee * abs(target - qty) * close[i]
                qty = target
        if combo.swing_leverage and not s_qty and entries[i] and cash > 1e-12:
            cash -= fee * combo.swing_leverage * cash
            margin = cash
            s_qty, s_entry, opened = combo.swing_leverage * margin / close[i], close[i], i
            trades += 1
        curve[i] = core + cash + s_qty * (close[i] - s_entry)
    final = curve[-1] - fee * s_qty * close[-1] - (fee * qty * close[-1] if perp else 0.0)
    curve[-1] = final
    return ComboResult(combo.name, float(final), cycles._drawdown(curve), core_dead, trades, lost, swept)


COMBOS = (
    Combo("Hold spot, all in", 1.0, 1.0, "spot"),
    Combo("Perp 2.5x held, all in", 1.0, 2.5, "held"),
    Combo("85% perp 2.5x held, 15% cash", 0.85, 2.5, "held"),
    Combo("85% perp 2.5x held + 15% swing 10x", 0.85, 2.5, "held", 10.0),
    Combo("85% perp 2.5x held + 15% swing 15x", 0.85, 2.5, "held", 15.0),
    Combo("90% perp 2.5x held + 10% swing 10x", 0.90, 2.5, "held", 10.0),
    Combo("80% perp 2.5x held + 20% swing 10x", 0.80, 2.5, "held", 10.0),
    Combo("85% perp 2.5x held + 10x swing, profit as margin", 0.85, 2.5, "held", 10.0, "margin"),
    Combo("85% perp kept at 2.5x, 15% cash", 0.85, 2.5, "kept"),
    Combo("85% perp kept at 2.5x + 15% swing 10x", 0.85, 2.5, "kept", 10.0),
    Combo("85% perp 1.4x held + 15% swing 10x", 0.85, 1.4, "held", 10.0),
    Combo("85% spot + 15% swing 10x", 0.85, 1.0, "spot", 10.0),
)


def run_window(full: pd.DataFrame, start, end, cfg: Config, rules: cycles.Rules | None = None,
               combos=COMBOS) -> list[ComboResult]:
    rules = rules or cycles.Rules.of(full)
    df = full[start:end]
    entries, exits = rules.cut("reclaim", df), rules.cut("overbought", df)
    return [simulate(c, df, cfg, entries, exits) for c in combos]


def chain(windows: list[list[ComboResult]]) -> list[ComboResult]:
    """Back to back, compounding: whatever is left at the end of one window starts the next with the same split."""
    out = []
    for rows in zip(*windows):
        out.append(ComboResult(
            rows[0].name, float(np.prod([r.multiple for r in rows])), max(r.max_drawdown for r in rows),
            next((r.core_liquidated for r in rows if r.core_liquidated is not None), None),
            sum(r.swing_trades for r in rows), sum(r.swing_liquidated for r in rows), sum(r.swept for r in rows)))
    return out


# ---- the arithmetic of leverage ------------------------------------------------------------------

@dataclass(frozen=True)
class Growth:
    """Annualised drift and volatility of daily log returns over a period."""
    label: str
    drift: float  # mean daily log return x 365: the growth rate of holding spot
    vol: float  # standard deviation of daily log returns x sqrt(365)

    def at(self, leverage: float, cost: float) -> float:
        """Long-run growth rate of a position kept at ``leverage``, paying ``cost`` a year on its notional:
        L(mu - cost) - L^2 sigma^2 / 2, with mu = drift + sigma^2 / 2. Holding spot (1x, no cost) grows at the drift."""
        mu = self.drift + self.vol ** 2 / 2
        return leverage * (mu - cost) - leverage ** 2 * self.vol ** 2 / 2

    def kelly(self, cost: float) -> float:
        """The leverage that maximises long-run growth; growth turns negative at twice this."""
        return (self.drift + self.vol ** 2 / 2 - cost) / self.vol ** 2


def growth(df: pd.DataFrame, label: str) -> Growth:
    returns = np.log(df["close"]).diff().dropna()
    return Growth(label, float(returns.mean() * 365), float(returns.std() * np.sqrt(365)))
