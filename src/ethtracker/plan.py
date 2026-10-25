"""The adding plan and the market context around it, as the dashboard shows them.

The plan is a long built up over time whose liquidation price is never allowed above a floor you choose.
A long's liquidation price depends only on the account's equity and the size held: equity + q(p - P) =
m q p. So the most it can hold at price P is equity / (P - floor(1 - m)), new cash can carry at most
P / (P - floor(1 - m)) leverage, and anything bought beyond that lifts the liquidation price above the floor.

The context follows what the research and the studies in this repository support: trend (the one return
signal with repeated out-of-sample support), volatility (forecastable, and what decides the odds of
reaching the floor), leverage crowding (a measure of fragility, not a timing signal), and the real hit
rates of "the pullback is over" signals, so none of them is mistaken for a high-confidence call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import cycles, indicators, pullbacks, swing

EWMA_LAMBDA = 0.94  # the RiskMetrics decay for daily volatility
LADDER_DEPTHS = (0.05, 0.09, 0.13, 0.20)
AT_FLOOR = 0.01  # a liquidation price within 1% of the floor counts as at it
ACTIVE_DAYS = 5
CROWDED, LIGHT = 0.90, 0.10  # funding percentiles over the last year
# A new plan's maintenance margin. Coinbase's ETH perp is Deribit's ETH-USDC linear perpetual, whose standard
# margin appears to be about 2% for maintenance (Deribit's Linear Perpetual help page, read via search summaries
# only). Assuming too little puts the real liquidation price above the floor; too much only buys slightly less.
PLAN_MMR = 0.02


def with_plan_margin(cfg, plan: dict | None):
    """The config with the plan's maintenance margin, so the CLI and the daily job size and mark alike."""
    if plan and plan.get("mmr") is not None:
        return replace(cfg, maintenance_margin_rate=plan["mmr"])
    return cfg


def _num(x) -> float | None:
    return None if x is None or not np.isfinite(x) else float(x)


# ---- sizing against the floor --------------------------------------------------------------------

def room(equity: float, qty: float, price: float, floor: float, mmr: float, cash: float = 0.0) -> float:
    """ETH that can be added with ``cash`` of new money while the liquidation price stays at or below ``floor``."""
    span = price - floor * (1 - mmr)
    if span <= 0:
        return 0.0
    return max(0.0, (equity + cash) / span - qty)


def sizing(cfg, plan: dict, price: float, equity: float, qty: float, liq: float | None,
           ladder_stats: dict[float, pullbacks.Rung] | None = None) -> dict:
    floor, weekly = plan["floor"], plan["weekly_add"]
    mmr = cfg.maintenance_margin_rate
    span = price - floor * (1 - mmr)
    cap = price / span if span > 0 else None
    weekly_qty = room(equity, qty, price, floor, mmr, weekly)
    status, cut_qty, margin_needed = "no position", 0.0, 0.0
    if qty > 0 and liq is not None:
        if liq > floor * (1 + AT_FLOOR):
            status = "above the floor"
            cut_qty = qty - max(0.0, equity / span) if span > 0 else qty
            margin_needed = qty * span - equity
        elif liq >= floor * (1 - AT_FLOOR):
            status = "at the floor"
        else:
            status = "below the floor"
    ladder = []
    for depth in LADDER_DEPTHS:
        level = price * (1 - depth)
        equity_there = equity + qty * (level - price)  # what the account would be worth if price fell to the level
        add = room(equity_there, qty, level, floor, mmr, weekly) if level > floor * (1 - mmr) else 0.0
        rung = (ladder_stats or {}).get(depth)
        ladder.append({
            "depth": depth, "price": level, "add_qty": add, "add_notional": add * level,
            "cap_leverage": level / (level - floor * (1 - mmr)) if level > floor * (1 - mmr) else None,
            "filled": _num(rung.filled) if rung else None, "held": _num(rung.held) if rung else None,
            "higher_365": _num(rung.higher_365) if rung else None,
        })
    return {
        "floor": floor, "weekly_add": weekly, "set_at": plan.get("set_at"),
        "status": status, "liq_price": liq, "equity": equity, "qty": qty,
        "cap_leverage": cap, "room_qty": room(equity, qty, price, floor, mmr),
        "weekly_qty": weekly_qty, "weekly_notional": weekly_qty * price,
        "cut_qty": cut_qty, "margin_needed": margin_needed,
        "ladder": ladder,
    }


# ---- volatility and the odds of reaching the floor --------------------------------------------------

def ewma_vol(close: pd.Series, lam: float = EWMA_LAMBDA) -> float | None:
    """Annualised RiskMetrics volatility forecast from daily log returns."""
    r = np.log(close).diff().dropna().to_numpy()
    if len(r) < 30:
        return None
    var = float(np.var(r[:30]))
    for x in r[30:]:
        var = lam * var + (1 - lam) * x * x
    return math.sqrt(var * indicators.DAYS_PER_YEAR)


def touch_probability(price: float, level: float, vol: float, years: float) -> float:
    """Chance a driftless random walk in log price touches ``level`` (below price) within ``years``:
    2 N(ln(level/price) / (vol sqrt(years))), the reflection principle. It says nothing about direction."""
    if level >= price:
        return 1.0
    z = math.log(level / price) / (vol * math.sqrt(years))
    return min(1.0, 2 * 0.5 * (1 + math.erf(z / math.sqrt(2))))


def floor_risk(history: pd.DataFrame, price: float, floor: float) -> dict:
    close = history["close"]
    vol = ewma_vol(close)
    realised = indicators.realized_vol(close, 30).iloc[-1]
    out = {
        "distance": price / floor - 1, "vol_forecast": vol, "vol_30d": _num(realised),
        "sigmas": math.log(price / floor) / vol if vol else None,
        "touch_30d": touch_probability(price, floor, vol, 30 / 365) if vol else None,
        "touch_1y": touch_probability(price, floor, vol, 1.0) if vol else None,
        "history": None,
    }
    depth = swing.today_depth(history)
    starts = cycles.deep_starts(history, depth=depth)
    if starts and floor < price:
        under = 1 - floor / price
        breaches = pullbacks.floor_breaches(history, starts, depths=(under,), years=(1, 4))[under]
        out["history"] = {"from_depth": depth, "starts": len(starts), "drawdowns": len(cycles.episodes(starts)),
                          "within_1y": breaches[1], "within_4y": breaches[4]}
    return out


# ---- trend, crowding, and the odds of the pullback signals ---------------------------------------------

def trend(history: pd.DataFrame) -> dict:
    close = history["close"]
    last = float(close.iloc[-1])
    sma50 = _num(indicators.sma(close, 50).iloc[-1])
    sma200 = _num(indicators.sma(close, 200).iloc[-1])
    r28 = _num(last / close.iloc[-29] - 1) if len(close) > 28 else None
    label = "Mixed"
    if sma50 and sma200 and r28 is not None:
        if last > sma200 and sma50 > sma200 and r28 > 0:
            label = "Uptrend"
        elif last < sma200 and sma50 < sma200 and r28 < 0:
            label = "Downtrend"
    return {"label": label, "close": last, "sma50": sma50, "sma200": sma200, "return_28d": r28,
            "above_50": sma50 is not None and last > sma50, "above_200": sma200 is not None and last > sma200}


def crowding(funding8h: pd.Series | None) -> dict | None:
    """Where funding sits against its own last year. High means crowded longs (fragile), not a short signal."""
    if funding8h is None or funding8h.dropna().empty:
        return None
    series = funding8h.dropna()
    now = float(series.iloc[-1])
    pct = float((series <= now).mean())
    label = "Crowded longs" if pct >= CROWDED else "Light longs" if pct <= LIGHT or now < 0 else "Normal"
    return {"funding_8h": now, "percentile": pct, "label": label, "days": len(series),
            "median_8h": float(series.median()), "high_8h": float(series.max())}


def odds(history: pd.DataFrame) -> dict:
    """Each pullback signal's real record: when it last fired and how often the low then held."""
    found = pullbacks.signals(history, None)
    pull = pullbacks.in_pullback(history)
    everyday = np.ones(len(history), dtype=bool)
    rows = []

    def row(name, mask, baseline=False):
        c = pullbacks.calibrate(name, history, pullbacks.at_close(history, mask))
        fired = np.flatnonzero(mask)
        last = None if baseline or not len(fired) else history.index[fired[-1]]
        rows.append({
            "name": name, "baseline": baseline, "events": c.n, "held": _num(c.held), "held_low": _num(c.held_low),
            "higher_365": _num(c.higher[365]), "last_fired": f"{last:%Y-%m-%d}" if last is not None else None,
            "days_ago": int((history.index[-1] - last).days) if last is not None else None,
            "active": last is not None and (history.index[-1] - last).days <= ACTIVE_DAYS,
        })

    row("Every day", everyday, baseline=True)
    row("Every pullback day", pull, baseline=True)
    for name, mask in found.items():
        row(name, mask)
    return {"in_pullback": bool(pull[-1]), "pullback_depth": pullbacks.PULLBACK, "further_fall": pullbacks.FURTHER,
            "held_days": pullbacks.HELD_DAYS, "signals": rows}


def ladder_stats(history: pd.DataFrame) -> dict[float, pullbacks.Rung]:
    """Fill rates and outcomes of orders under the close, from pullback days if ETH is in one, else every day."""
    placed_on = None if pullbacks.in_pullback(history)[-1] else np.ones(len(history), dtype=bool)
    return {r.depth: r for r in pullbacks.ladder(history, depths=LADDER_DEPTHS, placed_on=placed_on)}


# ---- buying now, or bidding lower, when every buy is sized to the floor ---------------------------------

@dataclass(frozen=True)
class Entry:
    depth: float  # 0 is buying at the close
    starts: int
    episodes: int
    filled: float  # share of orders that filled (the rest buy at the close when the order expires)
    eth: float  # median ETH bought per dollar, against buying at the close
    liquidated: float  # share whose floor was reached within the horizon: the same paths for every entry
    value: float  # median account value per dollar at the horizon, of those not liquidated
    beat_now: float  # share that ended worth more than buying at the close, of those not liquidated


def entry_vs_bid(full: pd.DataFrame, under: float, mmr: float, depths=LADDER_DEPTHS, days: int = pullbacks.BID_DAYS,
                 horizon: int = 365, placed_on: np.ndarray | None = None) -> list[Entry]:
    """From each start day: put one dollar to work with the liquidation price held at a floor ``under`` the close,
    either at that close or through an order ``depth`` under it, good for ``days`` (bought at the close if it never
    fills). Sized to the floor, a lower fill buys more ETH for the same dollar and the same liquidation price.
    Any fall to the floor passes through the order first, so both entries are liquidated on the same paths;
    values and wins are reported for the rest. Fees and funding are left out: both entries pay about the same."""
    close, low, open_ = (full[c].to_numpy(float) for c in ("close", "low", "open"))
    n = len(close)
    mask = pullbacks.in_pullback(full) if placed_on is None else np.asarray(placed_on, dtype=bool)
    starts = [i for i in np.flatnonzero(mask) if i + horizon < n]
    episodes = int(swing._first(mask)[: starts[-1] + 1].sum()) if starts else 0

    def outcome(i, j, price, floor):  # bought on day j at price; liquidated if a later low (or the fill day's) reaches the floor
        qty = 1.0 / (price - floor * (1 - mmr))
        end = i + horizon
        dead = low[j:end + 1].min() <= floor if j > i else low[i + 1:end + 1].min() <= floor
        return qty, 0.0 if dead else 1.0 + qty * (close[end] - price), dead

    now = {}
    for i in starts:
        floor = close[i] * (1 - under)
        now[i] = outcome(i, i, close[i], floor)
    alive = [i for i in starts if not now[i][2]]

    def median(values):
        return float(np.median(values)) if len(values) else float("nan")

    rows = [Entry(0.0, len(starts), episodes, 1.0, 1.0, float(np.mean([now[i][2] for i in starts])) if starts else float("nan"),
                  median([now[i][1] for i in alive]), float("nan"))]
    for depth in depths:
        fills, ratios, values, deaths, beats = 0, [], {}, [], {}
        for i in starts:
            floor, bid = close[i] * (1 - under), close[i] * (1 - depth)
            hit = np.flatnonzero(low[i + 1:i + days + 1] <= bid)
            if len(hit):
                j = i + 1 + int(hit[0])
                fills += 1
                qty, value, dead = outcome(i, j, min(bid, open_[j]), floor)
            else:
                j = i + days
                qty, value, dead = outcome(i, j, close[j], floor)
            ratios.append(qty / now[i][0])
            values[i] = value
            deaths.append(dead)
            beats[i] = value > now[i][1]
        rows.append(Entry(depth, len(starts), episodes, fills / len(starts) if starts else float("nan"),
                          float(np.median(ratios)) if starts else float("nan"), float(np.mean(deaths)) if starts else float("nan"),
                          median([values[i] for i in alive]), float(np.mean([beats[i] for i in alive])) if alive else float("nan")))
    return rows
