"""Per-position recommended action, with the bull and bear case behind it.

The rules are explicit and fully explained in ``reasons``. They are starting hypotheses, not
validated edge: the thresholds live in ``config.toml``. Actions are recommendations and alerts.
Nothing here places an order.

Priority, first match wins (for a long; a short mirrors it):
  1. Stop breached, or the time stop reached                      -> EXIT
  2. Whole book at or near liquidation                            -> EXIT
  3. Book outside the stress rule                                 -> TAKE_PARTIAL (cut size)
  4. Deep in the money and >= exit_risk_flags risk flags          -> EXIT
  5. Deep in the money and any risk flag or bearish net score     -> TAKE_PARTIAL + trail
  6. In profit, risk flags stacking and no net support            -> TAKE_PARTIAL + trail
  7. In profit and any risk flag or bearish net score             -> TIGHTEN the stop
  8. Deep in the money, signals supportive                        -> TIGHTEN (trail)
  9. Otherwise                                                    -> HOLD
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from . import indicators
from .config import Config
from .risk import Position, PositionStatus, _sign
from .signals import SignalSet

ACTIONS = ("HOLD", "TIGHTEN", "TAKE_PARTIAL", "EXIT")
PARTIAL_FRACTION = 0.5


@dataclass(frozen=True)
class Assessment:
    action: str
    reasons: tuple[str, ...]
    bull_case: tuple[str, ...]
    bear_case: tuple[str, ...]
    net_score: int  # family votes for minus against this position's direction (correlated signals vote once)
    risk_flags: int
    move: float  # signed price move in this position's favour since entry, fraction of entry
    deep_itm: bool
    suggested_stop: float | None  # only ever tighter than the current stop
    suggested_fraction: float  # share of the position to close


def trailing_stop(cfg: Config, pos: Position, df: pd.DataFrame, price: float) -> float | None:
    """A tighter stop than ``pos.stop`` if one exists: a chandelier trail, or breakeven once a full ATR in profit.

    Chandelier: ``trail_atr_multiple`` ATRs behind the best price since entry. Stops are never loosened,
    and a candidate on the wrong side of the current price is discarded.
    """
    s = _sign(pos.side)
    atr = indicators.atr(df, 14).iloc[-1]
    if pd.isna(atr):
        return None
    since = df[df.index >= pd.Timestamp(pos.opened_at).normalize()]
    best = price
    if not since.empty:
        best = max(price, float(since["high"].max())) if s == 1 else min(price, float(since["low"].min()))
    candidates = [best - s * cfg.trail_atr_multiple * float(atr)]
    if s * (price - pos.entry) >= float(atr):
        candidates.append(pos.entry)
    valid = [c for c in candidates if s * (price - c) > 0]
    tightest = max(valid + [pos.stop]) if s == 1 else min(valid + [pos.stop])
    return tightest if s * (tightest - pos.stop) > 0 else None


def assess(
    cfg: Config,
    pos: Position,
    status: PositionStatus,
    sigs: SignalSet,
    df: pd.DataFrame,
    portfolio_flags: Sequence[str] = (),
) -> Assessment:
    s = _sign(pos.side)
    price = status.price
    move = s * (price - pos.entry) / pos.entry
    deep = move * 100 >= cfg.deep_itm_pct

    aligned = [g for g in sigs.signals if g.bias == s]
    against = [g for g in sigs.signals if g.bias == -s]
    risky = [g for g in sigs.signals if g.risk]
    net = s * sum(sigs.votes().values())  # family votes, so correlated signals do not stack
    flags_n = len(risky)

    def tag(g) -> str:
        return g.headline + (" (unvalidated)" if g.unvalidated else "")

    bull = [tag(g) for g in aligned]
    bear = [tag(g) for g in against] + [tag(g) for g in risky if g not in against]
    if move > 0:
        bull.insert(0, f"In profit: {move:+.1%} from entry")
    if deep:
        bear.insert(0, f"Deep in the money ({move:+.1%}): the gain a reversal would take back")
    elif move < 0:
        bear.insert(0, f"Underwater: {move:+.1%} from entry")

    trail = trailing_stop(cfg, pos, df, price)
    book = set(portfolio_flags)
    reasons: list[str] = []
    action, fraction, wants_trail = "HOLD", 0.0, False

    if "STOP_HIT" in status.flags:
        action, fraction = "EXIT", 1.0
        reasons.append(f"Stop {pos.stop:,.2f} is breached")
    elif "TIME_STOP" in status.flags:
        action, fraction = "EXIT", 1.0
        reasons.append(f"Held {status.days_held:.0f} days: the {cfg.max_holding_days}-day time stop")
    elif book & {"LIQUIDATED", "LIQ_DANGER"}:
        action, fraction = "EXIT", 1.0
        reasons.append("The book is at or near liquidation")
    elif "STRESS_BREACH" in book:
        action, fraction, wants_trail = "TAKE_PARTIAL", PARTIAL_FRACTION, True
        reasons.append(f"The book no longer survives a {cfg.stress_drop_pct:g}% move: cut size")
    elif deep and flags_n >= cfg.exit_risk_flags:
        action, fraction = "EXIT", 1.0
        reasons.append(f"Deep in the money ({move:+.1%}) with {flags_n} risk flags (exit threshold {cfg.exit_risk_flags}): bank it")
    elif deep and (flags_n >= 1 or net < 0):
        action, fraction, wants_trail = "TAKE_PARTIAL", PARTIAL_FRACTION, True
        reasons.append(f"Deep in the money ({move:+.1%}) and the tide is turning ({flags_n} risk flag(s), net score {net:+d}): sell half, trail the rest")
    elif move > 0 and flags_n >= cfg.exit_risk_flags and net <= 0:
        action, fraction, wants_trail = "TAKE_PARTIAL", PARTIAL_FRACTION, True
        reasons.append(f"In profit with {flags_n} risk flags stacking and no net support (score {net:+d}): sell half")
    elif move > 0 and (flags_n >= 1 or net < 0):
        action, wants_trail = "TIGHTEN", True
        reasons.append(f"In profit and signals are softening ({flags_n} risk flag(s), net score {net:+d}): protect the gain")
    elif deep:
        action, wants_trail = "TIGHTEN", True
        reasons.append(f"Deep in the money ({move:+.1%}) with the bull case intact (score {net:+d}, no risk flags): trail the stop")
    else:
        if move <= 0 and flags_n >= cfg.exit_risk_flags and net < 0:
            reasons.append("Signals are deteriorating but the position is not in profit: honour the stop and do not add")
        else:
            reasons.append(f"No exit condition met (net score {net:+d}, {flags_n} risk flag(s))")

    return Assessment(
        action=action, reasons=tuple(reasons), bull_case=tuple(bull), bear_case=tuple(bear), net_score=net,
        risk_flags=flags_n, move=move, deep_itm=deep,
        suggested_stop=trail if wants_trail else None, suggested_fraction=fraction,
    )
