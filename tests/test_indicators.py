import numpy as np
import pandas as pd
import pytest

from ethtracker import indicators


def frame(close, high=None, low=None):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame({
        "open": close, "close": close,
        "high": close if high is None else pd.Series(high, dtype=float),
        "low": close if low is None else pd.Series(low, dtype=float),
        "volume": 1.0,
    })


def test_rsi_is_100_on_a_monotonic_rise_and_0_on_a_monotonic_fall():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert indicators.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert indicators.rsi(up[::-1].reset_index(drop=True)).iloc[-1] == pytest.approx(0.0)


def test_atr_of_a_constant_range():
    df = frame([100] * 30, high=[101] * 30, low=[99] * 30)
    assert indicators.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_realized_vol_annualises_daily_log_returns():
    close = pd.Series(100 * np.cumprod(np.where(np.arange(60) % 2 == 0, 1.01, 1 / 1.01)))
    daily = np.log(close).diff().iloc[-30:].std()
    assert indicators.realized_vol(close, 30).iloc[-1] == pytest.approx(daily * np.sqrt(365))
    flat_growth = pd.Series(100 * 1.01 ** np.arange(60))
    assert indicators.realized_vol(flat_growth, 30).iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_adverse_excursions_long_and_short():
    df = frame([100] * 5, high=[100, 100, 110, 100, 100], low=[100, 90, 100, 100, 100])
    assert list(indicators.adverse_excursions(df, "long", 2)) == pytest.approx([0.10, 0.0, 0.0])
    assert list(indicators.adverse_excursions(df, "short", 2)) == pytest.approx([0.10, 0.10, 0.0])


def test_adverse_excursions_is_empty_when_history_is_shorter_than_horizon():
    assert len(indicators.adverse_excursions(frame([100] * 5), "long", 10)) == 0


def test_snapshot_fields_and_price_override(candles):
    snap = indicators.snapshot(candles, price=2500.0)
    assert snap.price == 2500.0
    assert snap.sma_200 is not None and snap.rsi_14 is not None
    assert snap.atr_14_pct == pytest.approx(snap.atr_14 / 2500.0)
    assert snap.daily_sigma == pytest.approx(snap.vol_30d / np.sqrt(365))
    assert snap.drawdown_from_90d_high == pytest.approx(1 - 2500.0 / snap.high_90d)
    assert snap.candles == len(candles)


def test_snapshot_has_all_four_smas_and_volume(candles):
    snap = indicators.snapshot(candles)
    closes = candles["close"]
    for n in (20, 50, 100, 200):
        assert getattr(snap, f"sma_{n}") == pytest.approx(closes.tail(n).mean())
    assert snap.volume == 1000.0 and snap.volume_avg_20 == pytest.approx(1000.0)


def test_snapshot_volume_average_is_none_when_the_source_has_no_volume(candles):
    assert indicators.snapshot(candles.assign(volume=0.0)).volume_avg_20 is None


def test_frame_adds_the_requested_metrics_without_touching_the_candles(candles):
    out = indicators.frame(candles)
    assert list(out.columns[:5]) == list(candles.columns)
    for column in ("sma_20", "sma_50", "sma_100", "sma_200", "rsi_14", "atr_14", "vol_30d", "volume_sma_20"):
        assert column in out.columns
    assert out["sma_100"].iloc[-1] == pytest.approx(candles["close"].tail(100).mean())
    assert out["sma_200"].iloc[:199].isna().all() and not pd.isna(out["sma_200"].iloc[199])
    assert out["rsi_14"].iloc[-1] == pytest.approx(indicators.rsi(candles["close"]).iloc[-1])
    assert (out["open"] == candles["open"]).all()


def test_snapshot_pullback_gauges(candles):
    snap = indicators.snapshot(candles)
    month = candles.tail(30)
    assert snap.drawdown_from_30d_high == pytest.approx(1 - snap.price / month["high"].max())
    assert snap.rise_from_30d_low == pytest.approx(snap.price / month["low"].min() - 1)
    window = candles["close"].tail(20)
    assert snap.zscore_20 == pytest.approx((window.iloc[-1] - window.mean()) / window.std(ddof=0))


def test_snapshot_leaves_long_windows_empty_when_history_is_short(candles):
    snap = indicators.snapshot(candles.head(40))
    assert snap.sma_200 is None and snap.ret_90d is None
    assert snap.sma_20 is not None
