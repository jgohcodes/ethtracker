"""Each buy, with what the signals said when it was made and how it turned out.

A record is captured from the completed daily candles before the buy's UTC day, the same basis the dashboard
uses, so nothing from after the buy leaks in. It is stored with the buy in the journal and timestamped: the
buy's own time (``at``), when it was captured, and the close it was computed on, so it can be read back and
backtested against the same history later. The signal readings and headlines are stored as they were, not
recomputed, because the code that produces them will change.

The stance is a transparent score; every part is stored so it can be re-weighted later:

  trend            Uptrend +1, Downtrend -1           (time-series momentum: the best-supported signal)
  family vote      the sign of the summed family votes
  pullback signal  +1 if a "the pullback is over" signal fired within the last 5 days
  crowding         -1 if funding sits in its top 10% of the past year (crowded longs)
  risk flags       -1 if two or more technical risk flags are up

A total of +1 or more is "with the signals", -1 or less "against them", 0 "mixed". None of these is a
validated predictor (see pullback-study and signals-study): the point is to learn, over your own buys, which
were right and when going against them paid.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__, plan, signals

SCHEMA = 1
HORIZONS = (7, 30, 90, 365)
JUDGE_DAYS, EARLY_DAYS = 90, 30  # "were you right": higher this many days on, or after 30 days as an early read
FELL, RALLIED = 0.10, 0.15
SNAPSHOT_ROWS = 120  # newest buys shown on the dashboard; the summary covers all of them


def _sign(x: float) -> int:
    return int(x > 0) - int(x < 0)


def _round(x, digits=6):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def timestamp(value: str | None) -> pd.Timestamp:
    """A buy's time: ISO date or date-time, UTC unless it carries an offset; now if empty."""
    if not value:
        return pd.Timestamp(datetime.now(timezone.utc))
    at = pd.Timestamp(value)
    return at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")


def basis(history: pd.DataFrame, at: pd.Timestamp) -> pd.DataFrame:
    """The completed daily candles before the buy's UTC day."""
    return history[history.index < at.normalize()]


def capture(cfg, history: pd.DataFrame, at: pd.Timestamp, price: float, funding8h: pd.Series | None = None,
            floor: float | None = None) -> dict:
    closed = basis(history, at)
    if closed.empty:
        raise ValueError(f"no completed daily candles before {at:%Y-%m-%d}")
    sigs = signals.compute(cfg, closed)
    votes = sigs.votes()
    trend = plan.trend(closed)
    before = None if funding8h is None else funding8h[funding8h.index < at.normalize()]
    crowd = plan.crowding(before)
    odds = plan.odds(closed)
    tech = signals.technical_series(cfg, closed).iloc[-1]
    risk_flags = int(tech["risk_count"])
    active = [r["name"] for r in odds["signals"] if r["active"]]
    parts = {
        "trend": {"Uptrend": 1, "Downtrend": -1}.get(trend["label"], 0),
        "family_vote": _sign(sum(votes.values())),
        "pullback_signal": 1 if active else 0,
        "crowding": -1 if crowd and crowd["label"] == "Crowded longs" else 0,
        "risk_flags": -1 if risk_flags >= 2 else 0,
    }
    score = sum(parts.values())
    close = closed["close"]
    vol = plan.ewma_vol(close)
    record = {
        "v": SCHEMA, "at": at.isoformat(), "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "basis_close": f"{closed.index[-1]:%Y-%m-%d}", "code": __version__, "price": float(price),
        "stance": "with" if score > 0 else "against" if score < 0 else "mixed", "score": score, "parts": parts,
        "trend": trend["label"], "votes": votes, "risk_flags": risk_flags,
        "crowding": crowd["label"] if crowd else None, "in_pullback": odds["in_pullback"], "pullback_signals": active,
        "signals": [{"name": g.name, "bias": g.bias, "risk": g.risk, "headline": g.headline} for g in sigs.signals],
        "readings": {
            "close": _round(close.iloc[-1], 2), "sma50": _round(trend["sma50"], 2), "sma200": _round(trend["sma200"], 2),
            "return_28d": _round(trend["return_28d"], 4), "rsi": _round(tech["rsi"], 1), "z20": _round(tech["z20"], 2),
            "vol_30d": _round(tech["rv30"], 4), "vol_forecast": _round(vol, 4),
            "below_high": _round(1 - close.iloc[-1] / close.max(), 4),
            "below_30d_high": _round(1 - close.iloc[-1] / closed["high"].tail(30).max(), 4),
            "funding_8h": _round(crowd["funding_8h"], 8) if crowd else None,
            "funding_percentile": _round(crowd["percentile"], 3) if crowd else None,
        },
    }
    if floor is not None and vol:
        record["floor"] = {"floor": float(floor), "touch_30d": _round(plan.touch_probability(price, floor, vol, 30 / 365), 4),
                           "touch_1y": _round(plan.touch_probability(price, floor, vol, 1.0), 4)}
    return record


def outcome(history: pd.DataFrame, record: dict, price_now: float | None = None) -> dict:
    """What followed a buy, on the daily history: returns at each horizon from the buy price, the lowest low and
    the lowest close of the next 30 days, whether you were right, whether the signals were, and what they missed."""
    at = pd.Timestamp(record["at"])
    day, price, last = at.normalize(), record["price"], history.index[-1]
    close = history["close"]
    returns = {}
    for h in HORIZONS:
        target = day + pd.Timedelta(days=h)
        returns[h] = float(close[close.index <= target].iloc[-1] / price - 1) if target <= last else None
    window = history[(history.index > day) & (history.index <= day + pd.Timedelta(days=EARLY_DAYS))]
    low_30 = float(window["low"].min() / price - 1) if len(window) else None
    best_close_30 = float(window["close"].min() / price - 1) if len(window) else None
    judged_on = JUDGE_DAYS if returns[JUDGE_DAYS] is not None else EARLY_DAYS if returns[EARLY_DAYS] is not None else None
    judged = returns[judged_on] if judged_on else None
    score = record.get("score", 0)
    out = {
        "returns": returns, "now": (price_now if price_now else float(close.iloc[-1])) / price - 1,
        "days": int((last - day).days), "low_30": low_30, "best_close_30": best_close_30,
        # a 10% fall is known as soon as it happens; its absence only once the 30 days have passed
        "fell_10": True if low_30 is not None and low_30 <= -FELL else (False if returns[EARLY_DAYS] is not None else None),
        "judged_on": judged_on, "right": None if judged is None else judged > 0,
        "signals_right": None if judged is None or score == 0 else (score > 0) == (judged > 0),
        "misses": [], "unflagged": [],
    }
    early = returns[EARLY_DAYS]
    if early is not None:
        out["misses"] = [{"name": s["name"], "headline": s["headline"]} for s in record.get("signals", [])
                         if s["bias"] and _sign(s["bias"]) != _sign(early)]
        if low_30 is not None and low_30 <= -FELL and record.get("risk_flags", 0) == 0 and score >= 0:
            out["unflagged"].append(f"fell {-low_30:.0%} within 30 days with no risk flag up and the signals not against")
        if early >= RALLIED and score < 0:
            out["unflagged"].append(f"rose {early:.0%} in 30 days while the signals were against")
    return out


def summary(rows: list[dict]) -> dict:
    """Your record by stance, how often going against the signals paid, and each signal's hit rate on your buys."""
    by_stance = {}
    for stance in ("with", "mixed", "against"):
        group = [r for r in rows if r["stance"] == stance]
        judged = [r for r in group if r["outcome"]["right"] is not None]
        by_stance[stance] = {
            "buys": len(group), "judged": len(judged),
            "right": float(np.mean([r["outcome"]["right"] for r in judged])) if judged else None,
            "median_return": float(np.median([r["outcome"]["returns"][r["outcome"]["judged_on"]] for r in judged])) if judged else None,
        }
    hits: dict[str, list[bool]] = {}
    for r in rows:
        early = r["outcome"]["returns"][EARLY_DAYS]
        if early is None:
            continue
        for s in r.get("signals", []):
            if s["bias"]:
                hits.setdefault(s["name"], []).append(_sign(s["bias"]) == _sign(early))
    against = [r for r in rows if r["stance"] == "against" and r["outcome"]["right"] is not None]
    return {
        "buys": len(rows), "by_stance": by_stance,
        "against_and_right": sum(r["outcome"]["right"] for r in against), "against_judged": len(against),
        "unflagged": sum(len(r["outcome"]["unflagged"]) for r in rows),
        "signal_hits": sorted(({"name": k, "calls": len(v), "right": float(np.mean(v))} for k, v in hits.items()),
                              key=lambda x: (-x["calls"], x["name"])),
    }


def review(history: pd.DataFrame, decisions: list[dict], positions: dict, price_now: float | None = None) -> dict | None:
    """The dashboard's block: newest buys first, each with its outcome, and the summary over all of them."""
    rows = []
    for d in decisions:
        if "error" in d or "stance" not in d:
            continue
        pos = positions.get(d["position_id"])
        rows.append({**d, "qty": pos.qty if pos else d.get("qty"), "note": pos.note if pos else d.get("note", ""),
                     "outcome": outcome(history, d, price_now)})
    if not rows:
        return None
    rows.sort(key=lambda r: r["at"], reverse=True)
    lean = [{k: r[k] for k in ("position_id", "at", "basis_close", "price", "qty", "usd", "note", "stance", "score", "parts",
                               "trend", "risk_flags", "crowding", "in_pullback", "pullback_signals", "outcome") if k in r}
            for r in rows[:SNAPSHOT_ROWS]]
    return {"rows": lean, "summary": summary(rows), "missing": sum(1 for d in decisions if "error" in d)}
