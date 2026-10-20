import numpy as np
import pandas as pd
import pytest

from ethtracker import study
from ethtracker.config import Config
from tests.conftest import synthetic_candles

CFG = Config()  # fee 0.05%/side, funding 0.01%/8h, maintenance margin 0.5%
FEE, FUND = CFG.taker_fee_rate, CFG.funding_rate_8h


def bars(closes, highs=None, lows=None) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
    close = pd.Series(closes, dtype=float, index=idx)
    pick = lambda v: close if v is None else pd.Series(v, dtype=float, index=idx)
    return pd.DataFrame({"open": close, "high": pick(highs), "low": pick(lows), "close": close, "volume": 1.0})


def run(df, **overrides):
    params = dict(lookback=5, trigger=0.15, horizon=10, leverage=2, stop_pct=0.05, target_pct=0.10)
    params.update(overrides)
    return study.run(CFG, df, "long", **params)


def long_r(entry, exit_price, stop, days):
    loss = abs(entry - stop) + FEE * (entry + stop)
    pnl = exit_price - entry - FEE * (entry + exit_price) - entry * FUND * 3 * days
    return pnl / loss


def test_long_hits_target_and_r_is_net_of_fees_and_funding():
    df = bars([100] * 5 + [80, 85, 90, 96, 100])  # entry 80 on day 5 (20% below the 5d high of 100)
    (trade,) = run(df)
    assert trade.outcome == "target" and trade.days_held == 2
    assert (trade.entry, trade.exit) == (80, 88)
    assert trade.r == pytest.approx(long_r(80, 88, 76, 2))


def test_long_stopped_out_by_a_wick_just_over_minus_one_r():
    closes = [100] * 5 + [80, 82, 90, 96, 100]
    lows = list(closes)
    lows[6] = 75
    (trade,) = run(bars(closes, lows=lows))
    assert trade.outcome == "stop" and trade.exit == 76 and trade.days_held == 1
    assert trade.r == pytest.approx(long_r(80, 76, 76, 1))
    assert -1.01 < trade.r < -1.0


def test_stop_wins_when_one_day_touches_both_stop_and_target():
    closes = [100] * 5 + [80, 82, 90, 96, 100]
    highs, lows = list(closes), list(closes)
    highs[6], lows[6] = 95, 75
    (trade,) = run(bars(closes, highs=highs, lows=lows))
    assert trade.outcome == "stop"


def test_liquidation_before_a_wide_stop_loses_the_margin_not_the_stop():
    closes = [100] * 5 + [80, 82, 90, 96, 100]
    lows = list(closes)
    lows[6] = 70
    (trade,) = run(bars(closes, lows=lows), leverage=10, stop_pct=0.20)  # liq ~72.4 is nearer than the 64 stop
    assert trade.outcome == "liquidated"
    assert trade.r == pytest.approx(-(80 / 10) / (80 * 0.20 + FEE * (80 + 64)))


def test_time_stop_exits_at_the_close_of_the_last_allowed_day():
    (trade,) = run(bars([100] * 5 + [80] + [81] * 20), horizon=3)
    assert trade.outcome == "time" and trade.days_held == 3 and trade.exit == 81
    assert trade.r == pytest.approx(long_r(80, 81, 76, 3))


def test_triggers_during_an_open_trade_are_ignored():
    # day 6 (70) also sits >15% under the 5d high, but the day-5 trade is still open
    trades = run(bars([100] * 5 + [80, 70, 70, 70, 70, 70, 70]), horizon=3, stop_pct=0.5, target_pct=None)
    assert len(trades) == 1
    assert trades[0].entry_date == pd.Timestamp("2026-01-06", tz="UTC")


def test_short_mirrors_the_long_and_earns_funding_when_the_rate_is_positive():
    df = bars([100] * 5 + [120, 115, 110, 104, 100])
    (trade,) = study.run(CFG, df, "short", lookback=5, trigger=0.15, horizon=10, leverage=2, stop_pct=0.05, target_pct=0.10)
    assert trade.outcome == "target" and trade.days_held == 3 and trade.exit == pytest.approx(108)
    loss = 120 * 0.05 + FEE * (120 + 126)
    pnl = (120 - 108) - FEE * (120 + 108) + 120 * FUND * 3 * 3
    assert trade.r == pytest.approx(pnl / loss)


def test_atr_stops_scale_with_volatility_and_skip_bars_before_atr_exists():
    df = synthetic_candles(400)
    # a 5-day lookback would trigger within the first two weeks, before the 14d ATR exists
    trades = study.run(CFG, df, "long", lookback=5, trigger=0.03, horizon=14, leverage=3, stop_atr=2, target_r=2)
    assert trades and all(np.isfinite(t.r) for t in trades)
    assert all(t.entry_date >= df.index[13] for t in trades)
    assert all(t.exit_date > t.entry_date for t in trades)
    wide = study.run(CFG, df, "long", lookback=5, trigger=0.03, horizon=14, leverage=3, stop_atr=4, target_r=2)
    assert sum(t.outcome == "stop" for t in wide) < sum(t.outcome == "stop" for t in trades)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stop_pct=0.05, stop_atr=2),
        dict(),
        dict(stop_pct=0.05, target_pct=0.1, target_r=2),
        dict(stop_pct=0.05, side="sideways"),
    ],
)
def test_conflicting_or_missing_parameters_are_rejected(kwargs):
    side = kwargs.pop("side", "long")
    with pytest.raises(ValueError):
        study.run(CFG, bars([100] * 20), side, lookback=5, trigger=0.1, horizon=5, leverage=2, **kwargs)


def make_trades(rs):
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    return [study.Trade(ts, ts, "long", 100, 100, "target" if r > 0 else "stop", 2, r) for r in rs]


def test_summary_and_compounding():
    trades = make_trades([2, -1, -1, 3])
    summary = study.summarize(trades)
    assert (summary.n, summary.wins) == (4, 2)
    assert summary.avg_r == pytest.approx(0.75) and summary.total_r == 3
    assert summary.outcomes == {"target": 2, "stop": 2}
    total, drawdown = study.compound(trades, 10)
    assert total == pytest.approx(1.2 * 0.9 * 0.9 * 1.3 - 1)
    assert drawdown == pytest.approx(1 - 1.2 * 0.9 * 0.9 / 1.2)
    assert study.summarize([]) is None
