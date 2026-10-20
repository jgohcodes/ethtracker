"""Descriptive indicators on daily OHLC data. Facts about the tape, not buy/sell signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365  # crypto trades every day


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + gain / loss)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's average true range, in price units."""
    prev_close = df["close"].shift()
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def realized_vol(close: pd.Series, n: int = 30) -> pd.Series:
    """Annualised standard deviation of daily log returns."""
    return np.log(close).diff().rolling(n).std() * np.sqrt(DAYS_PER_YEAR)


def adverse_excursions(df: pd.DataFrame, side: str, horizon: int) -> np.ndarray:
    """Worst move against a position within ``horizon`` days of every possible entry.

    Entry is each day's close; the excursion uses the following days' lows (long)
    or highs (short), as a fraction of entry. Windows overlap heavily, so the
    effective sample is far smaller than the array length.
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    out = []
    for i in range(len(df) - horizon):
        window = slice(i + 1, i + 1 + horizon)
        if side == "long":
            out.append((close[i] - low[window].min()) / close[i])
        else:
            out.append((high[window].max() - close[i]) / close[i])
    return np.array(out)


SMA_WINDOWS = (20, 50, 100, 200)


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """Candles plus the full indicator time series, for export to another app."""
    out = df.copy()
    for n in SMA_WINDOWS:
        out[f"sma_{n}"] = sma(df["close"], n)
    out["rsi_14"] = rsi(df["close"], 14)
    out["atr_14"] = atr(df, 14)
    out["vol_30d"] = realized_vol(df["close"], 30)
    out["volume_sma_20"] = df["volume"].rolling(20).mean()
    return out


@dataclass(frozen=True)
class Snapshot:
    price: float
    ret_7d: float | None
    ret_30d: float | None
    ret_90d: float | None
    sma_20: float | None
    sma_50: float | None
    sma_100: float | None
    sma_200: float | None
    volume: float  # the latest candle, which is a partial day while the day is open
    volume_avg_20: float | None  # None when the source has no volume
    rsi_14: float | None
    atr_14: float | None
    atr_14_pct: float | None
    vol_30d: float | None  # annualised
    daily_sigma: float | None  # vol_30d / sqrt(365): a typical 1-day move
    high_90d: float
    low_90d: float
    drawdown_from_90d_high: float
    drawdown_from_30d_high: float  # how far below the recent high: the "undersold" gauge
    rise_from_30d_low: float  # how far above the recent low: the "overbought" gauge
    zscore_20: float | None  # std devs from the 20d mean
    candles: int


def _last(series: pd.Series) -> float | None:
    value = series.iloc[-1]
    return None if pd.isna(value) else float(value)


def _ret(close: pd.Series, n: int) -> float | None:
    return float(close.iloc[-1] / close.iloc[-1 - n] - 1) if len(close) > n else None


def with_price(df: pd.DataFrame, price: float | None) -> pd.DataFrame:
    """Copy of ``df`` whose latest candle reflects a fresher quote (close, and high/low if exceeded)."""
    df = df.copy()
    if price is not None:
        df.iloc[-1, df.columns.get_loc("close")] = price
        df.iloc[-1, df.columns.get_loc("high")] = max(df["high"].iloc[-1], price)
        df.iloc[-1, df.columns.get_loc("low")] = min(df["low"].iloc[-1], price)
    return df


def snapshot(df: pd.DataFrame, price: float | None = None) -> Snapshot:
    """Indicator summary. ``price`` replaces the last close when a fresher quote is available."""
    df = with_price(df, price)
    close = df["close"]
    last = float(close.iloc[-1])
    atr_14 = _last(atr(df, 14))
    vol = _last(realized_vol(close, 30))
    recent = df.tail(90)
    high_90 = float(recent["high"].max())
    month = df.tail(30)
    volume_avg = _last(df["volume"].rolling(20).mean())
    zscore =_last((close - sma(close, 20)) / close.rolling(20).std(ddof=0))
    return Snapshot(
        price=last,
        ret_7d=_ret(close, 7), ret_30d=_ret(close, 30), ret_90d=_ret(close, 90),
        sma_20=_last(sma(close, 20)), sma_50=_last(sma(close, 50)),
        sma_100=_last(sma(close, 100)), sma_200=_last(sma(close, 200)),
        volume=float(df["volume"].iloc[-1]),
        volume_avg_20=volume_avg or None,
        rsi_14=_last(rsi(close, 14)),
        atr_14=atr_14, atr_14_pct=None if atr_14 is None else atr_14 / last,
        vol_30d=vol, daily_sigma=None if vol is None else vol / np.sqrt(DAYS_PER_YEAR),
        high_90d=high_90, low_90d=float(recent["low"].min()),
        drawdown_from_90d_high=1 - last / high_90,
        drawdown_from_30d_high=1 - last / float(month["high"].max()),
        rise_from_30d_low=last / float(month["low"].min()) - 1,
        zscore_20=zscore,
        candles=len(df),
    )
