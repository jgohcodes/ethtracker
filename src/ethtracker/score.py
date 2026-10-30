"""One signal score from -1 to +1: the signals research supports, each weighted by how strong that support is.

The weights are set in advance from the literature, not fitted to ETH's history (fitting them would find
whatever worked by chance in one cycle):

  strong   time-series momentum and moving-average trend. Liu & Tsyvinski (2021, Review of Financial Studies)
           find strong time-series momentum in crypto at daily and weekly horizons, ETH included; Detzel, Liu,
           Strauss, Zhou & Zhu (2021, Financial Management) find price-to-moving-average ratios forecast bitcoin
           returns in and out of sample. Three readings of the same effect, on three horizons.
  weak     volume pressure and leverage crowding: plausible mechanisms, no peer-reviewed test on a single
           crypto asset that I could find. Small weights (0.125 each), so they can tip a close call and never
           outvote trend.
  none     RSI, extension, anchored VWAP, the volume profile, volatility, open interest: no published
           evidence of a directional edge, and this repository's own studies found none for oversold or
           "the dip is done" signals. Shown with their history, not scored.

``score-study`` checks the weights on ETH's history in two halves (before and from 2021): a component whose
direction is wrong in both is dropped, and the score's buckets give today's reading its meaning ("days scored
like today were followed by..."). The first run (ETH 2016-2026, 30 days ahead) found:

  momentum +0.12 / +0.12, trend +0.21 / +0.07, long trend +0.14 / +0.03, volume +0.03 / +0.09 (rank correlation
  with the next 30 days, to 2020 / from 2021); leverage crowding -0.02 / -0.06, wrong in both, so its weight is
  now 0 and the rest keep their proportions (0.30 : 0.25 : 0.20 : 0.125). Shuffling ETH's days gives about
  +/-0.08 by chance; the score's +0.14 overall is a modest edge, weaker since 2021.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators, signals

STRIP_DAYS = 30  # each signal's line: today and the 29 days before it
HISTORY_DAYS = 365  # the score against ETH's close: the page shows 3 months or the year
HORIZON = 30
FELL = 0.10
CROWDED = 0.90  # funding at or above this percentile of its past year
FUNDING_YEAR = 365
RECENT = "2021-01-01"  # the institutional era, reported on its own because the edge has been weaker since


@dataclass(frozen=True)
class Component:
    key: str
    name: str
    weight: float
    evidence: str  # strong, weak, or failed: pointed the wrong way on ETH's history
    reads: str


COMPONENTS = (
    Component("momentum", "Momentum, 7 and 28 days", 0.34, "strong",
              "+ when ETH is up over both the last 7 and 28 days, - when down over both, 0 when they disagree"),
    Component("trend", "Trend, 20 vs 50 days", 0.29, "strong",
              "+ when price and the 20-day average are above the 50-day, - when both below, else 0"),
    Component("long_trend", "Long trend, 200 days", 0.23, "strong", "+ above the 200-day average, - below"),
    Component("volume", "Volume pressure, 10 days", 0.14, "weak",
              f"- when down days carried {signals.VOLUME_PRESSURE_RATIO}x the volume of up days, + the reverse, else 0"),
    Component("crowding", "Leverage crowding", 0.0, "failed",
              f"- when funding is in the top {1 - CROWDED:.0%} of its past year, else 0. On ETH since 2019 crowded "
              "readings were followed by gains slightly more often, in both halves, so it carries no weight"),
)
WEIGHTS = {c.key: c.weight for c in COMPONENTS}

# what a score means, in words; the edges are fixed, not chosen from the data
BUCKETS = ((-1.01, -0.5, "Strongly negative"), (-0.5, -0.2, "Leaning negative"), (-0.2, 0.2, "Neutral"),
           (0.2, 0.5, "Leaning positive"), (0.5, 1.01, "Strongly positive"))


def bucket_of(value: float) -> tuple[float, float, str]:
    for lo, hi, name in BUCKETS:
        if lo < value <= hi:
            return lo, hi, name
    return BUCKETS[2]


def label(value: float) -> str:
    return bucket_of(value)[2]


def _daily_funding(index: pd.DatetimeIndex, funding8h: pd.Series | None) -> pd.Series | None:
    if funding8h is None or funding8h.dropna().empty:
        return None
    f = funding8h.dropna()
    f.index = f.index.normalize()
    return f.groupby(level=0).mean().reindex(index)


def _volume_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Volume on up days and on down days over the last 10, and whether 10 days of changes exist yet."""
    change = df["close"].diff()
    up = df["volume"].where(change > 0, 0.0).rolling(10).sum()
    down = df["volume"].where(change < 0, 0.0).rolling(10).sum()
    return up, down, change.rolling(10).count() >= 10


def _funding_percentile(index: pd.DatetimeIndex, funding8h: pd.Series | None) -> pd.Series | None:
    """Each day's funding as a percentile of its past year (NaN until 30 days are known)."""
    daily = _daily_funding(index, funding8h)
    if daily is None:
        return None

    def pct(window):
        last = window[-1]
        seen = window[~np.isnan(window)]
        return np.nan if np.isnan(last) or len(seen) < 30 else float((seen <= last).mean())
    return daily.rolling(FUNDING_YEAR, min_periods=1).apply(pct, raw=True)


def states(cfg, df: pd.DataFrame, funding8h: pd.Series | None = None) -> pd.DataFrame:
    """Each scored component's state for every day: +1, 0 or -1, NaN where it cannot be read yet.
    The trend, long-trend and volume states are the same rules as the dashboard's signals of those names."""
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    r7, r28 = close / close.shift(7) - 1, close / close.shift(28) - 1
    mom = np.where((r7 > 0) & (r28 > 0), 1.0, np.where((r7 < 0) & (r28 < 0), -1.0, 0.0))
    out["momentum"] = np.where(r28.isna(), np.nan, mom)

    s20, s50, s200 = (indicators.sma(close, w) for w in (20, 50, 200))
    trend = np.where((close > s50) & (s20 > s50), 1.0, np.where((close < s50) & (s20 < s50), -1.0, 0.0))
    out["trend"] = np.where(s50.isna(), np.nan, trend)
    out["long_trend"] = np.where(s200.isna(), np.nan, np.where(close > s200, 1.0, -1.0))

    up, down, ready = _volume_split(df)
    ratio = down / up.replace(0, np.nan)
    both = (up > 0) & (down > 0)
    r = signals.VOLUME_PRESSURE_RATIO
    vol = np.where(both & (ratio >= r), -1.0, np.where(both & (ratio <= 1 / r), 1.0, 0.0))
    out["volume"] = np.where(ready, vol, np.nan)

    p = _funding_percentile(df.index, funding8h)
    out["crowding"] = np.nan if p is None else np.where(p.isna(), np.nan, np.where(p >= CROWDED, -1.0, 0.0))
    return out


# What each scored component's line plots: the reading its rule thresholds, so a line drifting toward a
# threshold shows a state weakening before it flips. (key: unit, the line, a second line, where the state flips)
_R = signals.VOLUME_PRESSURE_RATIO
LINES = {
    "momentum": ("pct", "28-day return", "7-day return", (0.0,)),
    "trend": ("pct", "price vs the 50-day", "20-day vs the 50-day", (0.0,)),
    "long_trend": ("pct", "price vs the 200-day", None, (0.0,)),
    "volume": ("share", "up-day minus down-day share of 10-day volume", None, (-(_R - 1) / (_R + 1), (_R - 1) / (_R + 1))),
    "crowding": ("percentile", "funding's percentile of its past year", None, (CROWDED,)),
}


def readings(df: pd.DataFrame, funding8h: pd.Series | None = None) -> pd.DataFrame:
    """The continuous readings behind each scored state, one column per line in ``LINES`` (``key`` and
    ``key_2`` for a second line). Their signs against the thresholds give the states."""
    close = df["close"]
    s20, s50, s200 = (indicators.sma(close, w) for w in (20, 50, 200))
    up, down, ready = _volume_split(df)
    p = _funding_percentile(df.index, funding8h)
    return pd.DataFrame({
        "momentum": close / close.shift(28) - 1, "momentum_2": close / close.shift(7) - 1,
        "trend": close / s50 - 1, "trend_2": s20 / s50 - 1,
        "long_trend": close / s200 - 1,
        "volume": ((up - down) / (up + down).replace(0, np.nan)).where(ready),
        "crowding": np.nan if p is None else p,
    }, index=df.index)


VOL_DAYS = 30  # daily volatility for scaling the continuous parts


def continuous(df: pd.DataFrame, funding8h: pd.Series | None = None) -> pd.DataFrame:
    """The scored parts on a sliding scale from -1 to +1 instead of a step: each reading divided by how far it
    normally moves (ETH's daily volatility over the reading's horizon), then squashed with tanh. Same signs as
    ``states``; a part that is negative and getting more so reads lower each day instead of sitting at -1."""
    rd = readings(df, funding8h)
    vol = np.log(df["close"]).diff().rolling(VOL_DAYS).std()
    scale = lambda days: vol * np.sqrt(days)  # noqa: E731
    out = pd.DataFrame(index=df.index)
    out["momentum"] = (np.tanh(rd["momentum"] / scale(28)) + np.tanh(rd["momentum_2"] / scale(7))) / 2
    out["trend"] = (np.tanh(rd["trend"] / scale(25)) + np.tanh(rd["trend_2"] / scale(15))) / 2
    out["long_trend"] = np.tanh(rd["long_trend"] / scale(100))
    _, hi = LINES["volume"][3]
    out["volume"] = np.tanh(rd["volume"] / (2 * hi))
    out["crowding"] = np.where(rd["crowding"].isna(), np.nan, -np.clip((rd["crowding"] - 0.5) / (CROWDED - 0.5), 0, 1))
    return out.where(rd[out.columns].notna())


def unscored_lines(cfg) -> dict[str, tuple[str, str, tuple[float, ...]]]:
    """The same for the unscored signals: unit, what the line is, and where the signal's reading flips."""
    return {
        "rsi": ("index", "RSI, 14 days", (cfg.rsi_oversold, cfg.rsi_overbought)),
        "hammer": ("pct", "lower wick as a share of the day's range", (0.6,)),
        "rsi_rollover": ("index", "RSI, 14 days", (cfg.rsi_overbought,)),
        "extension": ("z", "std devs from the 20-day mean", (-cfg.extension_z, cfg.extension_z)),
        "volatility": ("ratio", "the higher of 7d/30d realised vol and ATR/its 180d median", (cfg.vol_spike_ratio,)),
        "avwap_low": ("pct", "price vs the AVWAP", (0.0,)),
        "avwap_year": ("pct", "price vs the AVWAP", (0.0,)),
        "poc": ("pct", "price vs the point of control", (0.0,)),
        "funding": ("rate8h", "funding per 8 hours", (-cfg.funding_hot_8h, cfg.funding_hot_8h)),
        "oi_change": ("pct", "open interest, 24-hour change", (-cfg.oi_spike_pct / 100, cfg.oi_spike_pct / 100)),
        "oi_level": ("percentile", "open interest's percentile of its history", (cfg.oi_high_percentile / 100,)),
        "long_liquidations": ("z", "std devs above the 60-day mean", (cfg.liq_spike_z,)),
        "short_liquidations": ("z", "std devs above the 60-day mean", (cfg.liq_spike_z,)),
    }


def _value(v) -> float | None:
    return None if v is None or pd.isna(v) else float(f"{float(v):.6g}")


def combine(st: pd.DataFrame, weights: dict[str, float] = WEIGHTS) -> pd.Series:
    """The weighted average of the states that can be read that day: a missing component (no funding history
    yet) drops out and the rest are rescaled, so the score always runs from -1 to +1."""
    w = pd.Series(weights)
    present = st[w.index].notna()
    total = (st[w.index].fillna(0) * w).sum(axis=1)
    weight = (present * w).sum(axis=1)
    return (total / weight.replace(0, np.nan)).where(st[["momentum", "trend", "long_trend"]].notna().all(axis=1))


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    """Spearman rank correlation (ties averaged), without scipy."""
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def shuffled(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The same days in a random order: each day's return, range and volume kept together, any pattern across
    days destroyed. Scores on these show how large an IC chance alone produces."""
    close = df["close"]
    rows = np.column_stack([np.log(close).diff().fillna(0).to_numpy(), (df["low"] / close).to_numpy(),
                            (df["high"] / close).to_numpy(), df["volume"].to_numpy()])
    rows = rows[rng.permutation(len(rows))]
    c = float(close.iloc[0]) * np.exp(np.cumsum(rows[:, 0]))
    return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * rows[:, 2], "low": c * rows[:, 1], "close": c,
                         "volume": rows[:, 3]}, index=df.index)


def chance_ic(cfg, df: pd.DataFrame, weights: dict[str, float] = WEIGHTS, runs: int = 40, horizon: int = HORIZON,
              seed: int = 0) -> float:
    """The spread (one standard deviation) of the score's IC when the days are shuffled: a no-edge yardstick."""
    rng = np.random.default_rng(seed)
    ics = []
    for _ in range(runs):
        fake = shuffled(df, rng)
        sc, fwd = combine(states(cfg, fake), weights), forward(fake, horizon)
        m = sc.notna() & fwd["ret"].notna()
        ics.append(rank_ic(sc[m], fwd.loc[m, "ret"]))
    return float(np.nanstd(ics))


def forward(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    close, low = df["close"], df["low"]
    ahead = close.shift(-horizon) / close - 1
    lows = low[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)  # lowest low of the next `horizon` days
    return pd.DataFrame({"ret": ahead, "fell": (lows / close - 1) <= -FELL + 1e-9}, index=df.index).where(ahead.notna())


def calibration(df: pd.DataFrame, score: pd.Series, value: float, horizon: int = HORIZON) -> dict:
    """What followed days whose score fell in today's bucket, against every day: ETH's own record."""
    fwd = forward(df, horizon)
    lo, hi, name = bucket_of(value)
    known = fwd["ret"].notna() & score.notna()
    same = known & (score > lo) & (score <= hi)

    def stats(mask):
        n = int(mask.sum())
        if not n:
            return {"days": 0, "episodes": 0, "higher": None, "median": None, "fell": None}
        return {"days": n, "episodes": signals.count_episodes(mask), "higher": float((fwd.loc[mask, "ret"] > 0).mean()),
                "median": float(fwd.loc[mask, "ret"].median()), "fell": float(fwd.loc[mask, "fell"].astype(bool).mean())}

    recent = df.index >= pd.Timestamp(RECENT, tz="UTC")
    return {"bucket": name, "range": [max(lo, -1.0), min(hi, 1.0)], "horizon": horizon, "same": stats(same),
            "all": stats(known), "same_recent": stats(same & recent), "all_recent": stats(known & recent),
            "from": f"{df.index[0]:%Y-%m-%d}", "recent_from": RECENT}


def block(cfg, history: pd.DataFrame, funding8h: pd.Series | None, frame: pd.DataFrame, derivs: pd.DataFrame | None) -> dict:
    """The dashboard's signal score: today's value and label, its 90-day path, what similar scores led to, every
    scored component with its weight, 30-day line and daily states, and the unscored signals the same way."""
    st = states(cfg, history, funding8h)
    rd = readings(history, funding8h).tail(STRIP_DAYS)
    ct = continuous(history, funding8h)
    step = combine(st)
    score = combine(ct)  # the score shown: each part on its sliding scale, so it moves inside a step
    today = float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else None
    path = score.tail(HISTORY_DAYS)
    ago = {str(k): (float(score.iloc[-1 - k]) if len(score) > k and pd.notna(score.iloc[-1 - k]) else None) for k in (3, 7, 14)}
    components = []
    for c in COMPONENTS:
        strip = st[c.key].tail(STRIP_DAYS)
        now = strip.iloc[-1]
        unit, name, second, flips = LINES[c.key]
        components.append({
            "key": c.key, "name": c.name, "weight": c.weight, "evidence": c.evidence, "reads": c.reads,
            "state": None if pd.isna(now) else int(now),
            "contribution": None if pd.isna(ct[c.key].iloc[-1]) or today is None else float(ct[c.key].iloc[-1] * c.weight / sum(
                w for k, w in WEIGHTS.items() if pd.notna(ct[k].iloc[-1]))),
            "strip": [None if pd.isna(v) else int(v) for v in strip],
            "line": {"unit": unit, "label": name, "values": [_value(v) for v in rd[c.key]], "thresholds": list(flips),
                     "second": {"label": second, "values": [_value(v) for v in rd[f"{c.key}_2"]]} if second else None},
        })
    return {
        "value": today, "label": label(today) if today is not None else None, "ago": ago,
        "history": [{"date": f"{d:%Y-%m-%d}", "score": round(float(v), 4), "close": round(float(history["close"][d]), 2)}
                    for d, v in path.items() if pd.notna(v)],
        "step": None if pd.isna(step.iloc[-1]) else float(step.iloc[-1]),
        "calibration": calibration(history, score, today) if today is not None else None,
        "components": components, "unscored": unscored(cfg, frame, derivs),
        "strip_dates": [f"{d:%Y-%m-%d}" for d in history.index[-STRIP_DAYS:]],
    }


SCORED_ELSEWHERE = {"trend", "long_trend", "volume_pressure"}  # the same rules as the trend, long_trend and volume components


def at_times(cfg, history: pd.DataFrame, funding8h: pd.Series | None, derivs: pd.DataFrame | None, times) -> list[dict | None]:
    """The score and the signals as they stood when each trade was made: on the completed daily closes before
    the trade's UTC day, the same basis as the buy log, so nothing from the trade's own day leaks in."""
    st = states(cfg, history, funding8h)
    sc = combine(continuous(history, funding8h))
    out = []
    for at in times:
        if not at:
            out.append(None)
            continue
        at = pd.Timestamp(at)
        day = (at.tz_localize("UTC") if at.tzinfo is None else at.tz_convert("UTC")).normalize()
        before = history[history.index < day]
        if len(before) < 2:
            out.append(None)
            continue
        d = before.index[-1]
        sigs = signals.compute(cfg, before, None if derivs is None else derivs[derivs.index < day])
        value = sc.get(d)
        out.append({
            "basis_close": f"{d:%Y-%m-%d}", "close": float(before["close"].iloc[-1]),
            "score": None if pd.isna(value) else float(value), "label": None if pd.isna(value) else label(value),
            "components": [{"key": c.key, "name": c.name, "weight": c.weight,
                            "state": None if pd.isna(st[c.key].get(d)) else int(st[c.key][d])} for c in COMPONENTS],
            "firing": [{"name": g.name, "bias": g.bias, "risk": g.risk, "headline": g.headline}
                       for g in sigs.signals if (g.bias or g.risk) and g.name not in SCORED_ELSEWHERE],
        })
    return out


def unscored(cfg, frame: pd.DataFrame, derivs: pd.DataFrame | None) -> list[dict]:
    """Every other signal, recomputed as it read on each of the last 30 days, for its line."""
    rows: dict[str, dict] = {}
    lines = unscored_lines(cfg)
    n = len(frame)
    for back in range(STRIP_DAYS - 1, -1, -1):
        sub = frame.iloc[: n - back]
        if len(sub) < 2:
            continue
        cut = sub.index[-1] + pd.Timedelta(days=1)
        sigs = signals.compute(cfg, sub, None if derivs is None else derivs[derivs.index < cut])
        for g in sigs.signals:
            if g.name in SCORED_ELSEWHERE:
                continue
            row = rows.setdefault(g.name, {"name": g.name, "strip": [None] * STRIP_DAYS, "values": [None] * STRIP_DAYS,
                                           "risks": [None] * STRIP_DAYS})
            row["strip"][STRIP_DAYS - 1 - back] = g.bias
            row["values"][STRIP_DAYS - 1 - back] = _value(g.value)
            row["risks"][STRIP_DAYS - 1 - back] = g.risk
            if back == 0:
                row.update(state=g.bias, risk=g.risk, headline=g.headline, unvalidated=g.unvalidated)
    out = []
    for r in rows.values():
        if "headline" not in r:
            continue
        values, risks = r.pop("values"), r.pop("risks")
        if r["name"] in lines and any(v is not None for v in values):
            unit, name, flips = lines[r["name"]]
            r["line"] = {"unit": unit, "label": name, "values": values, "thresholds": list(flips), "second": None, "risks": risks}
        out.append(r)
    return out


# ---- the score against price: echo, lead, and how late it turns -------------------------------------

LEADS = (1, 3, 7, 14, 30, 60, 90, 180)  # days ahead
AGES = (0, 3, 7, 14, 30, 60)  # how old a score is when acted on
ECHOES = (7, 28, 90)  # days back
SWING = 0.30  # a turning point is a high or low ETH then moved 30% away from


def _ahead(close: pd.Series, days: int) -> pd.Series:
    return close.shift(-days) / close - 1


def lead(close: pd.Series, score: pd.Series, horizons=LEADS, mask: pd.Series | None = None) -> dict[int, float]:
    """Rank correlation of today's score with the return over the next ``h`` days, for each ``h``."""
    out = {}
    for h in horizons:
        fwd = _ahead(close, h)
        m = score.notna() & fwd.notna() & (True if mask is None else mask)
        out[h] = rank_ic(score[m], fwd[m])
    return out


def stale(close: pd.Series, score: pd.Series, ages=AGES, horizon: int = HORIZON) -> dict[int, float]:
    """Acting on a score ``k`` days old: its rank correlation with the next ``horizon`` days from today."""
    fwd = _ahead(close, horizon)
    out = {}
    for k in ages:
        old = score.shift(k)
        m = old.notna() & fwd.notna()
        out[k] = rank_ic(old[m], fwd[m])
    return out


def echo(close: pd.Series, score: pd.Series, horizons=ECHOES) -> dict[int, float]:
    """Rank correlation of today's score with the return over the past ``h`` days: how much it repeats price."""
    out = {}
    for h in horizons:
        back = close / close.shift(h) - 1
        m = score.notna() & back.notna()
        out[h] = rank_ic(score[m], back[m])
    return out


def chance_leads(cfg, df: pd.DataFrame, runs: int = 30, seed: int = 1) -> dict[int, float]:
    """The spread of each forward rank correlation with ETH's days shuffled: the no-edge yardstick per horizon."""
    rng = np.random.default_rng(seed)
    seen: dict[int, list[float]] = {h: [] for h in LEADS}
    for _ in range(runs):
        fake = shuffled(df, rng)
        for h, ic in lead(fake["close"], combine(states(cfg, fake))).items():
            seen[h].append(ic)
    return {h: float(np.nanstd(v)) for h, v in seen.items()}


def pivots(close: pd.Series, swing: float = SWING) -> list[tuple[pd.Timestamp, str, float]]:
    """Turning points on closes: a high is confirmed once price falls ``swing`` below it, a low once price rises
    ``swing`` above it (hindsight, for measuring lags only)."""
    out: list[tuple[pd.Timestamp, str, float]] = []
    hi_t = lo_t = close.index[0]
    hi = lo = float(close.iloc[0])
    direction = 0  # +1 looking for the next high, -1 for the next low
    for t, p in close.items():
        p = float(p)
        if direction >= 0 and p > hi:
            hi, hi_t = p, t
        if direction <= 0 and p < lo:
            lo, lo_t = p, t
        if direction >= 0 and p <= hi * (1 - swing):
            out.append((hi_t, "top", hi))
            direction, lo, lo_t = -1, p, t
        elif direction <= 0 and p >= lo * (1 + swing):
            out.append((lo_t, "bottom", lo))
            direction, hi, hi_t = 1, p, t
    return [x for x in out if x[0] != close.index[0]]  # the first day is where the data starts, not a turn


def turns(close: pd.Series, score: pd.Series, swing: float = SWING) -> list[dict]:
    """At each turning point: when the score crossed zero the new way (days after; negative = before), the price
    then, and how much of the swing to the next turning point was still to come."""
    points = pivots(close, swing)
    rows = []
    for k, (t, kind, price) in enumerate(points):
        want = (score < 0) if kind == "top" else (score > 0)
        at = score.index.get_loc(t)
        if pd.isna(score.iloc[at]):
            continue
        if want.iloc[at]:  # already flipped: find when, looking back
            before = (~want.iloc[: at + 1]) & score.iloc[: at + 1].notna()
            idx = np.flatnonzero(before.to_numpy())
            flip = idx[-1] + 1 if len(idx) else 0
        else:
            ahead = np.flatnonzero(want.iloc[at:].to_numpy())
            flip = at + int(ahead[0]) if len(ahead) else None
        flip_t = score.index[flip] if flip is not None else None
        flip_price = float(close.iloc[flip]) if flip is not None else None
        nxt = points[k + 1] if k + 1 < len(points) else None
        prev = points[k - 1] if k > 0 else None
        if flip_t is None or (nxt and flip_t >= nxt[0]) or (prev and flip_t <= prev[0]):
            # the score never turned for this swing: it sat out a bounce in a falling market (or a dip in a rising
            # one), already the right way round for the next turn
            rows.append({"date": t, "kind": kind, "price": price, "flip": None, "lag": None, "at_flip": None,
                         "swing": (nxt[2] if nxt else float(close.iloc[-1])) / price - 1, "complete": nxt is not None,
                         "left": None})
            continue
        end_price = nxt[2] if nxt else float(close.iloc[-1])
        swing_size = end_price / price - 1
        rows.append({
            "date": t, "kind": kind, "price": price, "flip": flip_t, "lag": int((flip_t - t).days),
            "at_flip": flip_price / price - 1, "swing": swing_size, "complete": nxt is not None,
            "left": (end_price - flip_price) / (end_price - price) if end_price != price else float("nan"),
        })
    return rows


def flips_per_year(score: pd.Series) -> float:
    s = np.sign(score.dropna())
    s = s[s != 0]
    changes = int((s != s.shift()).sum()) - 1
    years = (s.index[-1] - s.index[0]).days / 365.25
    return changes / years if years > 0 else float("nan")


# ---- trading the score's crossings ---------------------------------------------------------------------

@dataclass(frozen=True)
class CrossRun:
    multiple: float  # final value / starting cash
    max_drawdown: float
    in_market: float  # share of days holding any ETH
    buys: int


def cross_strategy(close: pd.Series, sc: pd.Series, sell: float = 1.0, take_profit: float | None = None,
                   band: float = 0.0, fee: float = 0.0005) -> CrossRun:
    """Spot ETH, all cash in when the score crosses above +``band``; when it crosses below -``band``, sell the share
    ``sell`` of the ETH held. With ``take_profit``, also sell everything once the close is that far above the average
    buy price. Trades at the signal day's close (the score uses that close), with ``fee`` a side."""
    c, s = close.to_numpy(float), sc.to_numpy(float)
    cash, eth, cost = 1.0, 0.0, 0.0
    peak, mdd, held, buys = 1.0, 0.0, 0, 0
    armed = None  # +1 after an up cross, -1 after a down cross
    for i in range(1, len(c)):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            continue
        if s[i] > band and armed != 1:
            if cash > 0:
                q = cash * (1 - fee) / c[i]
                cost += cash
                eth, cash, buys = eth + q, 0.0, buys + 1
            armed = 1
        elif s[i] < -band and armed != -1:
            if eth > 0:
                q = eth * sell
                cash += q * c[i] * (1 - fee)
                cost *= (1 - sell)
                eth -= q
            armed = -1
        if take_profit is not None and eth > 0 and cost > 0 and eth * c[i] >= cost * (1 + take_profit):
            cash += eth * c[i] * (1 - fee)
            eth, cost = 0.0, 0.0
        value = cash + eth * c[i]
        peak = max(peak, value)
        mdd = max(mdd, 1 - value / peak)
        held += eth > 0
    final = cash + eth * c[-1]
    return CrossRun(final, mdd, held / max(len(c) - 1, 1), buys)


def hold(close: pd.Series) -> CrossRun:
    c = close.to_numpy(float)
    v = c / c[0]
    return CrossRun(float(v[-1]), float((1 - v / np.maximum.accumulate(v)).max()), 1.0, 1)
