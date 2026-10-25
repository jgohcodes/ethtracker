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

STRIP_DAYS = 15  # today and the 14 days before it
HISTORY_DAYS = 90
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


def states(cfg, df: pd.DataFrame, funding8h: pd.Series | None = None) -> pd.DataFrame:
    """Each scored component's state for every day: +1, 0 or -1, NaN where it cannot be read yet.
    The trend, long-trend and volume states are the same rules as the dashboard's signals of those names."""
    close, volume = df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)

    r7, r28 = close / close.shift(7) - 1, close / close.shift(28) - 1
    mom = np.where((r7 > 0) & (r28 > 0), 1.0, np.where((r7 < 0) & (r28 < 0), -1.0, 0.0))
    out["momentum"] = np.where(r28.isna(), np.nan, mom)

    s20, s50, s200 = (indicators.sma(close, w) for w in (20, 50, 200))
    trend = np.where((close > s50) & (s20 > s50), 1.0, np.where((close < s50) & (s20 < s50), -1.0, 0.0))
    out["trend"] = np.where(s50.isna(), np.nan, trend)
    out["long_trend"] = np.where(s200.isna(), np.nan, np.where(close > s200, 1.0, -1.0))

    change = close.diff()
    up = volume.where(change > 0, 0.0).rolling(10).sum()
    down = volume.where(change < 0, 0.0).rolling(10).sum()
    ratio = down / up.replace(0, np.nan)
    both = (up > 0) & (down > 0)
    r = signals.VOLUME_PRESSURE_RATIO
    vol = np.where(both & (ratio >= r), -1.0, np.where(both & (ratio <= 1 / r), 1.0, 0.0))
    out["volume"] = np.where(change.rolling(10).count() < 10, np.nan, vol)

    daily = _daily_funding(df.index, funding8h)
    if daily is None:
        out["crowding"] = np.nan
    else:
        def pct(window):
            last = window[-1]
            seen = window[~np.isnan(window)]
            return np.nan if np.isnan(last) or len(seen) < 30 else float((seen <= last).mean())
        p = daily.rolling(FUNDING_YEAR, min_periods=1).apply(pct, raw=True)
        out["crowding"] = np.where(p.isna(), np.nan, np.where(p >= CROWDED, -1.0, 0.0))
    return out


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
    scored component with its weight and 15-day strip, and the unscored signals with their strips."""
    st = states(cfg, history, funding8h)
    score = combine(st)
    today = float(score.iloc[-1]) if pd.notna(score.iloc[-1]) else None
    path = score.tail(HISTORY_DAYS)
    ago = {str(k): (float(score.iloc[-1 - k]) if len(score) > k and pd.notna(score.iloc[-1 - k]) else None) for k in (3, 7, 14)}
    components = []
    for c in COMPONENTS:
        strip = st[c.key].tail(STRIP_DAYS)
        now = strip.iloc[-1]
        components.append({
            "key": c.key, "name": c.name, "weight": c.weight, "evidence": c.evidence, "reads": c.reads,
            "state": None if pd.isna(now) else int(now),
            "contribution": None if pd.isna(now) or today is None else float(now * c.weight / sum(
                w for k, w in WEIGHTS.items() if pd.notna(st[k].iloc[-1]))),
            "strip": [None if pd.isna(v) else int(v) for v in strip],
        })
    return {
        "value": today, "label": label(today) if today is not None else None, "ago": ago,
        "history": [{"date": f"{d:%Y-%m-%d}", "score": round(float(v), 4)} for d, v in path.items() if pd.notna(v)],
        "calibration": calibration(history, score, today) if today is not None else None,
        "components": components, "unscored": unscored(cfg, frame, derivs),
        "strip_dates": [f"{d:%Y-%m-%d}" for d in history.index[-STRIP_DAYS:]],
    }


SCORED_ELSEWHERE = {"trend", "long_trend", "volume_pressure"}  # the same rules as the trend, long_trend and volume components


def unscored(cfg, frame: pd.DataFrame, derivs: pd.DataFrame | None) -> list[dict]:
    """Every other signal, recomputed as it read on each of the last 15 days, for its strip."""
    rows: dict[str, dict] = {}
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
            row = rows.setdefault(g.name, {"name": g.name, "strip": [None] * STRIP_DAYS})
            row["strip"][STRIP_DAYS - 1 - back] = g.bias
            if back == 0:
                row.update(state=g.bias, risk=g.risk, headline=g.headline, unvalidated=g.unvalidated)
    return [r for r in rows.values() if "headline" in r]


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
