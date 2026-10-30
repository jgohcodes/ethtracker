"""Market signals: technical overlays and derivatives positioning.

A signal's ``bias`` is its direction for *price* (+1 bullish, -1 bearish, 0 neutral). Its ``risk``
flag marks a condition that raises the odds of a violent move in either direction, which is what a
leveraged position should be most wary of. These are conditions, not predictions. ``unvalidated``
marks signals with no historical test behind them in this tool (the derivatives ones: free history
of open interest and liquidations is too short or absent).

Anchored VWAP and the volume profile are built from daily candles, so they are coarser than the
tick or intraday versions: each day's volume is spread evenly across its high-low range.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import indicators
from .config import Config

ROLLOVER_POINTS = 10  # RSI must fall this far from its 10d high (while that high was overbought)
VOLUME_PRESSURE_RATIO = 1.3
MIN_OI_PERCENTILE_DAYS = 30
MIN_LIQ_DAYS = 16
STUDY_WARMUP = 60  # candles before every indicator in technical_series is defined


@dataclass(frozen=True)
class Signal:
    name: str
    bias: int
    risk: bool
    headline: str
    unvalidated: bool = False
    family: str = ""  # correlated signals share a family and vote once; empty means its own family
    value: float | None = None  # the reading the rule thresholds (RSI, std devs, % from a level...), for its line


# "below the 20d, 50d, 200d, both AVWAPs and the POC" is one fact seen six ways, so those vote once
FAMILIES = {
    "trend": "levels", "long_trend": "levels", "avwap_low": "levels", "avwap_year": "levels", "poc": "levels",
    "extension": "momentum", "rsi": "momentum", "rsi_rollover": "momentum",
    "volume_pressure": "volume",
    "funding": "derivatives", "oi_change": "derivatives", "long_liquidations": "derivatives",
}


@dataclass(frozen=True)
class VolumeProfile:
    poc: float  # price level with the most traded volume
    vah: float  # value-area high
    val: float  # value-area low


@dataclass(frozen=True)
class SignalSet:
    price: float
    signals: tuple[Signal, ...]
    levels: dict
    notes: tuple[str, ...]

    def get(self, name: str) -> Signal | None:
        return next((s for s in self.signals if s.name == name), None)

    def votes(self) -> dict[str, int]:
        """Each family's direction for price: +1, -1 or 0, from the sign of its signals' summed bias."""
        sums: dict[str, int] = {}
        for g in self.signals:
            key = g.family or g.name
            sums[key] = sums.get(key, 0) + g.bias
        return {key: (total > 0) - (total < 0) for key, total in sums.items()}


def _f(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def technical_series(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Per-day technical metrics and the three technical risk flags (stretched, RSI hot, volatility spike)."""
    close = df["close"]
    z20 = (close - close.rolling(20).mean()) / close.rolling(20).std(ddof=0)
    rsi = indicators.rsi(close, 14)
    rsi_peak = rsi.rolling(ROLLOVER_POINTS).max()
    logret = np.log(close).diff()
    rv7 = logret.rolling(7).std() * np.sqrt(indicators.DAYS_PER_YEAR)
    rv30 = logret.rolling(30).std() * np.sqrt(indicators.DAYS_PER_YEAR)
    atr = indicators.atr(df, 14)
    atr_pct = atr / close
    out = pd.DataFrame({
        "z20": z20,
        "rsi": rsi,
        "rsi_peak": rsi_peak,
        "rv7": rv7,
        "rv30": rv30,
        "vol_ratio": rv7 / rv30.replace(0, np.nan),
        "atr_ratio": atr_pct / atr_pct.rolling(180, min_periods=60).median(),
        "range_ratio": (df["high"] - df["low"]) / atr.shift(1),
    })
    out["ext_hot"] = out["z20"] >= cfg.extension_z
    out["rsi_hot"] = (out["rsi"] >= cfg.rsi_overbought) | (
        (out["rsi_peak"] >= cfg.rsi_overbought) & (out["rsi"] < out["rsi_peak"] - ROLLOVER_POINTS)
    )
    out["vol_spike"] = (
        (out["vol_ratio"] >= cfg.vol_spike_ratio)
        | (out["atr_ratio"] >= cfg.vol_spike_ratio)
        | (out["range_ratio"] >= cfg.range_spike_atr)
    )
    out["risk_count"] = out[["ext_hot", "rsi_hot", "vol_spike"]].sum(axis=1)
    return out


def anchored_vwap(df: pd.DataFrame, anchor: pd.Timestamp) -> float | None:
    """Volume-weighted average typical price from ``anchor`` to the latest candle."""
    sub = df[df.index >= anchor]
    volume = sub["volume"]
    if sub.empty or volume.sum() <= 0:
        return None
    typical = (sub["high"] + sub["low"] + sub["close"]) / 3
    return float((typical * volume).sum() / volume.sum())


def volume_profile(df: pd.DataFrame, lookback: int = 365, bins: int = 100, value_area: float = 0.70) -> VolumeProfile | None:
    """Point of control and value area over the last ``lookback`` candles."""
    sub = df.tail(lookback)
    volume = sub["volume"].to_numpy(float)
    if len(sub) < 20 or volume.sum() <= 0:
        return None
    lo, hi = float(sub["low"].min()), float(sub["high"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    lows, highs = sub["low"].to_numpy(), sub["high"].to_numpy()
    overlap = np.clip(np.minimum(edges[None, 1:], highs[:, None]) - np.maximum(edges[None, :-1], lows[:, None]), 0, None)
    totals = overlap.sum(axis=1)
    flat = totals <= 0  # a zero-range candle: its volume goes in the bin holding its price
    weights = np.divide(overlap, totals[:, None], out=np.zeros_like(overlap), where=~flat[:, None])
    if flat.any():
        cols = np.clip(np.searchsorted(edges, lows[flat], side="right") - 1, 0, bins - 1)
        weights[np.flatnonzero(flat), cols] = 1.0
    profile = (weights * volume[:, None]).sum(axis=0)

    poc = int(profile.argmax())
    lo_i = hi_i = poc
    held, target = profile[poc], value_area * profile.sum()
    while held < target and (lo_i > 0 or hi_i < bins - 1):
        below = profile[lo_i - 1] if lo_i > 0 else -1.0
        above = profile[hi_i + 1] if hi_i < bins - 1 else -1.0
        if above >= below:
            hi_i += 1
            held += profile[hi_i]
        else:
            lo_i -= 1
            held += profile[lo_i]
    centre = (edges[:-1] + edges[1:]) / 2
    return VolumeProfile(poc=float(centre[poc]), vah=float(edges[hi_i + 1]), val=float(edges[lo_i]))


def _technical(cfg: Config, df: pd.DataFrame, anchor: pd.Timestamp | None, signals: list, levels: dict, notes: list) -> None:
    close = df["close"]
    price = float(close.iloc[-1])
    n = len(df)
    sma = {w: _f(indicators.sma(close, w).iloc[-1]) for w in indicators.SMA_WINDOWS}
    levels.update({f"sma_{w}": v for w, v in sma.items() if v is not None})
    piv = indicators.monthly_pivots(df)
    if piv:
        levels.update(piv)
    levels.update(indicators.nearest_swings(df, price))

    if sma[20] is not None and sma[50] is not None:
        if price > sma[50] and sma[20] > sma[50]:
            signals.append(Signal("trend", 1, False, f"Uptrend: price above a 20d ({sma[20]:,.0f}) that sits above the 50d ({sma[50]:,.0f})"))
        elif price < sma[50] and sma[20] < sma[50]:
            signals.append(Signal("trend", -1, False, f"Downtrend: price below a 20d ({sma[20]:,.0f}) that sits below the 50d ({sma[50]:,.0f})"))
        else:
            signals.append(Signal("trend", 0, False, f"Mixed trend: price {price:,.0f}, 20d {sma[20]:,.0f}, 50d {sma[50]:,.0f}"))
    else:
        notes.append(f"trend signals need 50 daily candles (have {n})")
    if sma[200] is not None:
        above = price > sma[200]
        signals.append(Signal("long_trend", 1 if above else -1, False, f"{'Above' if above else 'Below'} the 200d ({sma[200]:,.0f}), {price / sma[200] - 1:+.1%}"))
    else:
        notes.append(f"the 200d trend signal needs 200 daily candles (have {n})")

    row = technical_series(cfg, df).iloc[-1]
    z = _f(row["z20"])
    if z is not None:
        if row["ext_hot"]:
            signals.append(Signal("extension", -1, True, f"Stretched: {z:+.1f} std devs above the 20d mean", value=z))
        elif z <= -cfg.extension_z:
            signals.append(Signal("extension", 1, False, f"Far below the 20d mean: {z:+.1f} std devs", value=z))
        else:
            signals.append(Signal("extension", 0, False, f"Not stretched: {z:+.1f} std devs from the 20d mean", value=z))
    rsi = _f(row["rsi"])
    if rsi is not None:
        bias = -1 if rsi >= cfg.rsi_overbought else (1 if rsi <= cfg.rsi_oversold else 0)
        word = {-1: f"at or over {cfg.rsi_overbought:.0f}", 1: f"at or under {cfg.rsi_oversold:.0f}", 0: "neutral"}[bias]
        signals.append(Signal("rsi", bias, bool(row["rsi_hot"]), f"RSI {rsi:.0f}: {word}", value=rsi))
        peak = _f(row["rsi_peak"])
        if peak is not None and peak >= cfg.rsi_overbought and rsi < peak - ROLLOVER_POINTS:
            signals.append(Signal("rsi_rollover", -1, False, f"Momentum rolling over: RSI 10d high {peak:.0f}, now {rsi:.0f}", value=rsi))

    vol_ratio, atr_ratio, range_ratio = (_f(row[k]) for k in ("vol_ratio", "atr_ratio", "range_ratio"))
    if vol_ratio is not None or atr_ratio is not None:
        parts = []
        if vol_ratio is not None:
            parts.append(f"7d/30d realised vol {vol_ratio:.2f}x ({row['rv7']:.0%} vs {row['rv30']:.0%})")
        if atr_ratio is not None:
            parts.append(f"ATR {atr_ratio:.2f}x its 180d median")
        if range_ratio is not None:
            parts.append(f"latest range {range_ratio:.1f}x ATR")
        state = "Volatility spiking" if row["vol_spike"] else "Volatility normal"
        signals.append(Signal("volatility", 0, bool(row["vol_spike"]), f"{state}: {'; '.join(parts)}",
                              value=max(v for v in (vol_ratio, atr_ratio) if v is not None)))
    else:
        notes.append("volatility signals need about 40 daily candles")

    if df["volume"].sum() > 0:
        anchors = {}
        low_at = anchor if anchor is not None else df["low"].tail(365).idxmin()
        anchors["avwap_low"] = (low_at, "the chosen anchor" if anchor is not None else "the 365d low")
        this_year = df.index[df.index.year == df.index[-1].year]
        if len(this_year):
            anchors["avwap_year"] = (this_year[0], "the year open")
        for name, (when, label) in anchors.items():
            v = anchored_vwap(df, when)
            if v is not None:
                levels[name] = v
                above = price > v
                signals.append(Signal(name, 1 if above else -1, False, f"{'Above' if above else 'Below'} AVWAP from {label} ({when:%Y-%m-%d}): ${v:,.0f}, {price / v - 1:+.1%}",
                                      value=price / v - 1))
        prof = volume_profile(df)
        if prof is not None:
            levels.update(poc=prof.poc, vah=prof.vah, val=prof.val)
            above = price > prof.poc
            signals.append(Signal("poc", 1 if above else -1, False, f"{'Above' if above else 'Below'} the 365d point of control (${prof.poc:,.0f}); value area ${prof.val:,.0f} to ${prof.vah:,.0f}",
                                  value=price / prof.poc - 1))
        else:
            notes.append("volume profile needs 20 candles with volume")

        last10 = df.tail(11)
        change = last10["close"].diff().iloc[1:]
        vol10 = last10["volume"].iloc[1:]
        up, down = float(vol10[change > 0].sum()), float(vol10[change < 0].sum())
        if up > 0 and down > 0:
            ratio = down / up
            if ratio >= VOLUME_PRESSURE_RATIO:
                signals.append(Signal("volume_pressure", -1, False, f"Selling volume: down days carried {ratio:.1f}x the volume of up days over 10d"))
            elif ratio <= 1 / VOLUME_PRESSURE_RATIO:
                signals.append(Signal("volume_pressure", 1, False, f"Buying volume: up days carried {1 / ratio:.1f}x the volume of down days over 10d"))
            else:
                signals.append(Signal("volume_pressure", 0, False, f"Volume balanced between up and down days ({ratio:.1f}x down/up over 10d)"))
    else:
        notes.append("the candle source has no volume: anchored VWAP, volume profile and volume pressure are unavailable")


def _derivatives(cfg: Config, derivs: pd.DataFrame | None, price_change: float | None, signals: list, notes: list) -> None:
    if derivs is None or derivs.empty:
        notes.append("no derivatives data: open-interest, funding and liquidation signals are unavailable (run `ethtracker record`, or pass --derivs-csv)")
        return

    if "funding_8h" in derivs and derivs["funding_8h"].notna().any():
        f = float(derivs["funding_8h"].dropna().iloc[-1])
        hot = cfg.funding_hot_8h
        if f >= hot:
            signals.append(Signal("funding", -1, True, f"Funding {f:.3%} per 8h (>= {hot:.3%}): longs paying heavily, crowded", True, value=f))
        elif f <= -hot:
            signals.append(Signal("funding", 1, True, f"Funding {f:.3%} per 8h (<= -{hot:.3%}): shorts paying heavily, squeeze risk", True, value=f))
        else:
            signals.append(Signal("funding", 0, False, f"Funding {f:.3%} per 8h: not extreme (threshold {hot:.3%})", True, value=f))
    else:
        notes.append("no funding data")

    if "open_interest" in derivs and derivs["open_interest"].notna().any():
        oi = derivs["open_interest"].dropna()
        latest = float(oi.iloc[-1])
        reference = oi[oi.index <= oi.index[-1] - pd.Timedelta(hours=24)]
        if reference.empty:
            hours = (oi.index[-1] - oi.index[0]).total_seconds() / 3600
            notes.append(f"open-interest change needs 24h of history (have {hours:.0f}h)")
        else:
            change = latest / float(reference.iloc[-1]) - 1
            thr = cfg.oi_spike_pct / 100
            up = price_change is not None and price_change > 0
            move = "" if price_change is None else f" with price {price_change:+.1%}"
            if change >= thr and price_change is not None:
                text, bias = (f"OI {change:+.1%}{move}: new longs driving the move", 1) if up else (f"OI {change:+.1%}{move}: new shorts pressing, or longs trapped", -1)
            elif change <= -thr and price_change is not None:
                text, bias = (f"OI {change:+.1%}{move}: short covering, a weak rally", 0) if up else (f"OI {change:+.1%}{move}: leverage being flushed out", 0)
            else:
                text, bias = f"OI {change:+.1%} in 24h{move}: no spike (threshold {thr:.0%})", 0
            signals.append(Signal("oi_change", bias, False, text, True, value=change))
        daily = oi.resample("1D").last().dropna()
        if len(daily) >= MIN_OI_PERCENTILE_DAYS:
            pctile = float((daily <= latest).mean() * 100)
            crowded = pctile >= cfg.oi_high_percentile
            text = f"OI at the {pctile:.0f}th percentile of its {len(daily)}-day history" + (": crowded" if crowded else "")
            signals.append(Signal("oi_level", 0, crowded, text, True, value=pctile / 100))
        else:
            notes.append(f"the open-interest percentile needs {MIN_OI_PERCENTILE_DAYS} days of history (have {len(daily)})")
    else:
        notes.append("no open-interest data")

    for column, label in (("long_liq_usd", "Long"), ("short_liq_usd", "Short")):
        if column not in derivs or not derivs[column].notna().any():
            continue
        daily = derivs[column].resample("1D").sum(min_count=1).dropna()
        if len(daily) < MIN_LIQ_DAYS:
            notes.append(f"{label.lower()}-liquidation spikes need {MIN_LIQ_DAYS} days of data (have {len(daily)})")
            continue
        history = daily.iloc[:-1].tail(60)
        std = float(history.std(ddof=0))
        if std <= 0:
            continue
        z = (float(daily.iloc[-1]) - float(history.mean())) / std
        spike = z >= cfg.liq_spike_z
        if label == "Long":
            text = f"Long liquidations spiked (z = {z:.1f}): forced selling" if spike else f"Long liquidations normal (z = {z:.1f})"
            signals.append(Signal("long_liquidations", 1 if spike else 0, False, text, True, value=z))
        else:
            text = f"Short liquidations spiked (z = {z:.1f}): a squeeze is under way, momentum but exhaustion risk" if spike else f"Short liquidations normal (z = {z:.1f})"
            signals.append(Signal("short_liquidations", 0, spike, text, True, value=z))


def compute(cfg: Config, df: pd.DataFrame, derivs: pd.DataFrame | None = None, anchor: pd.Timestamp | None = None) -> SignalSet:
    """All signals at the latest candle. Pass candles whose last row already reflects the live price."""
    signals: list[Signal] = []
    levels: dict[str, float] = {}
    notes: list[str] = []
    _technical(cfg, df, anchor, signals, levels, notes)
    price_change = float(df["close"].iloc[-1] / df["close"].iloc[-2] - 1) if len(df) >= 2 else None
    _derivatives(cfg, derivs, price_change, signals, notes)
    signals = [replace(g, family=FAMILIES.get(g.name, g.name)) for g in signals]
    return SignalSet(price=float(df["close"].iloc[-1]), signals=tuple(signals), levels=levels, notes=tuple(notes))


def count_episodes(mask: pd.Series) -> int:
    """Separate runs of consecutive True days: a 3-day flagged stretch is one episode, not three."""
    mask = mask.astype(bool)
    return int((mask & ~mask.shift(fill_value=False)).sum())


@dataclass(frozen=True)
class Bucket:
    label: str
    n: int
    episodes: int
    median_return: float
    drawdown_15: float  # share of entries that saw a >= 15% adverse move within the horizon
    drawdown_20: float
    rally_10: float  # share that ended the horizon >= 10% higher: the upside an early exit gives up


def forward_outcomes(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """For a long entered at each day's close: return at the horizon and the worst intra-horizon drawdown."""
    close, low = df["close"].to_numpy(), df["low"].to_numpy()
    rows = []
    for i in range(len(df) - horizon):
        window = slice(i + 1, i + 1 + horizon)
        rows.append((close[i + horizon] / close[i] - 1, (close[i] - low[window].min()) / close[i]))
    return pd.DataFrame(rows, columns=["fwd_return", "drawdown"], index=df.index[: len(rows)])


def warning_study(cfg: Config, df: pd.DataFrame, horizon: int = 14) -> list[Bucket]:
    """How the technical risk-flag count related to what followed: all days, then by count (0, 1, 2, 3).

    Only the three candle-derived flags are testable. Open interest, funding and liquidations have no
    free history long enough to test, so they are excluded here and stay unvalidated.
    """
    outcomes = forward_outcomes(df, horizon)
    counts = technical_series(cfg, df)["risk_count"].reindex(outcomes.index)
    # before the indicators exist every flag is False, which would pollute the "0 flags" bucket
    warmed_up = pd.Series(np.arange(len(outcomes)) >= STUDY_WARMUP, index=outcomes.index)

    def bucket(label: str, mask) -> Bucket | None:
        selected = mask & warmed_up
        sub = outcomes[selected]
        if sub.empty:
            return None
        return Bucket(label, len(sub), count_episodes(selected), float(sub["fwd_return"].median()), float((sub["drawdown"] >= 0.15).mean()),
                      float((sub["drawdown"] >= 0.20).mean()), float((sub["fwd_return"] >= 0.10).mean()))

    buckets = [bucket("all days", pd.Series(True, index=outcomes.index))]
    for k in range(4):
        buckets.append(bucket(f"{k} flag{'s' if k != 1 else ''}", counts == k))
    buckets.append(bucket("2+ flags", counts >= 2))
    return [b for b in buckets if b is not None]
