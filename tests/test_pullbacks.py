import numpy as np
import pandas as pd
import pytest

from ethtracker import derivs, pullbacks
from ethtracker.cli import main
from ethtracker.config import Config

from tests.conftest import synthetic_candles
from tests.test_cycles import cycle_path, flags, path


def bars(closes, lows=None, opens=None, highs=None):
    df = path(closes, lows=lows)
    if opens is not None:
        df["open"] = opens
    if highs is not None:
        df["high"] = highs
    return df


def test_a_capitulation_wick_is_a_spike_down_that_closes_in_the_top_half():
    closes = [100.0] * 30 + [84.0, 85.0]
    lows = [100.0] * 30 + [84.0, 76.0]  # day 31 trades 10.6% under the previous close
    highs = [100.0] * 30 + [100.0, 86.0]  # day 30 opened at 100 and closed on its low
    df = bars(closes, lows=lows, highs=highs)
    assert list(np.flatnonzero(pullbacks.capitulation_wick(df))) == [31]
    df.loc[df.index[31], "close"] = 77.0  # closes near its low: not bought, not a wick
    assert not pullbacks.capitulation_wick(df).any()


def test_negative_funding_only_counts_during_a_pullback():
    df = bars([100.0] * 30 + [80.0, 80.0, 80.0])
    funding = pd.Series(0.0001, index=df.index)
    funding.iloc[[5, 31]] = -0.0001
    assert list(np.flatnonzero(pullbacks.funding_negative(df, funding))) == [31]
    assert pullbacks.funding_negative(df, None) is None


def test_confluence_needs_two_different_signals_within_five_days():
    a, b = flags(20, 3), flags(20, 6)
    assert list(np.flatnonzero(pullbacks.confluence([a, b]))) == [6]
    assert not pullbacks.confluence([flags(20, 3), flags(20, 9)]).any()
    assert not pullbacks.confluence([flags(20, 3, 6)]).any()  # one signal twice is not confluence


def test_orders_fill_at_their_price_or_the_open_when_price_gaps_through():
    closes = [100.0] * 30 + [85.0, 84.0, 70.0, 80.0]
    lows = [100.0] * 30 + [85.0, 74.0, 70.0, 80.0]
    opens = [100.0] * 30 + [85.0, 84.0, 72.0, 80.0]
    df = bars(closes, lows=lows, opens=opens)
    # pullback starts on day 30 (85 is 15% under 100): the order sits at 74.8, fills on day 31 at 74.8
    assert pullbacks.bid_fills(df) == [(31, pytest.approx(85 * 0.88))]
    opens[31] = 74.0  # opens below the order: filled at the open
    assert pullbacks.bid_fills(bars(closes, lows=lows, opens=opens)) == [(31, 74.0)]
    lows[31], lows[32] = 80.0, 80.0
    assert pullbacks.bid_fills(bars([100.0] * 30 + [85.0, 84.0, 80.0, 80.0], lows=lows, opens=opens)) == []


def test_the_lower_bound_is_honest_about_small_samples():
    assert pullbacks.wilson_low(8, 10) == pytest.approx(0.49, abs=0.01)
    assert pullbacks.wilson_low(80, 100) == pytest.approx(0.71, abs=0.01)
    assert np.isnan(pullbacks.wilson_low(0, 0))


def test_calibration_counts_the_low_holding_and_price_higher():
    closes = [100.0] * 400
    lows = [100.0] * 400
    lows[5] = 85.0  # a 15% fall the day after entry 1 at day 4
    df = bars(closes, lows=lows)
    c = pullbacks.calibrate("x", df, [(4, 100.0), (10, 95.0)])
    assert c.n == 2 and c.held == 0.5  # the first entry's low did not hold; the second's did
    assert c.higher[30] == 0.5 and c.higher[365] == 0.5 and c.n_year == 2  # only the entry at 95 ended higher
    filled = pullbacks.calibrate("x", df, [(5, 95.0)], from_fill=True)
    assert filled.held == 0.0  # an order filled during day 5 sees that day's low of 85


def test_the_ladder_fills_more_often_the_closer_the_order():
    rungs = pullbacks.ladder(synthetic_candles(n=900, seed=4))
    fills = [r.filled for r in rungs]
    assert fills == sorted(fills, reverse=True) and rungs[0].days > 0 and rungs[0].episodes > 0


def test_adding_weekly_and_saving_for_a_signal():
    closes = [100.0] * 7 + [50.0] * 7 + [100.0]
    df = bars(closes)
    signal = np.full(15, np.nan)
    signal[10] = 50.0
    dca = pullbacks.accumulate("x", df, 0.0, np.full(15, np.nan), 1.0)
    assert dca.multiple == pytest.approx((1 / 100 + 1 / 50 + 1 / 100) * 100 / 3) and dca.idle == 0.0
    saved = pullbacks.accumulate("x", df, 0.0, signal, 0.0)
    # two units saved by day 10 buy at 50; the third (day 14) is still cash at the end
    assert saved.multiple == pytest.approx((2 / 50 * 100 + 1) / 3) and saved.idle == pytest.approx(1 / 3)


def test_pullback_study_command(tmp_path, capsys, monkeypatch):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.9).to_numpy(), start="2016-05-18")
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "prices.csv", index=False)
    monkeypatch.setattr(derivs, "fetch_funding_history",
                        lambda start, end, **kw: (pd.Series(np.where(np.arange(len(pd.date_range(start, end))) % 9 == 0, -0.0001, 0.0001),
                                                            index=pd.date_range(start, end, freq="D", tz="UTC")), 0))
    assert main(["--provider", "csv", "--csv", str(tmp_path / "prices.csv"), "pullback-study"]) == 0
    text = capsys.readouterr().out
    for heading in ("Last fired", "How often the low held", "as it is today", "A buy order under the close",
                    "Adding one unit every week", "like today", "Every week (DCA)", "Funding flips negative"):
        assert heading in text, heading


# ---- buying every week, with a few days' choice ------------------------------------------------------

def test_the_weekly_buy_takes_the_first_qualifying_day_or_the_last():
    df = bars([100.0, 101, 99, 98, 97, 96, 95] + [90.0, 91, 92, 93, 94, 95, 96])
    good = np.array([False, False, True, True, False, False, False] + [False] * 7)
    assert list(pullbacks.window_buy_days(df, good)) == [2, 10]  # week 2 has no qualifying day: its 4th day
    assert list(pullbacks.window_buy_days(df, None)) == [0, 7]
    assert list(pullbacks.window_buy_days(df, "best")) == [3, 7] and list(pullbacks.window_buy_days(df, "worst")) == [1, 10]


def test_hindsight_bounds_every_rule():
    df = synthetic_candles(n=800, seed=13)
    rows = {r.name: r.multiple for r in pullbacks.Timing(df, Config()).run(df, df.index[0], df.index[-1], Config())}
    best, worst = rows.pop("Lowest close of the 4 days (hindsight)"), rows.pop("Highest close of the 4 days (hindsight)")
    assert all(worst <= m <= best for m in rows.values())


def test_floor_breaches_count_the_lows_after_each_start():
    df = path([100.0] * 800, lows=[100.0] * 400 + [55.0] + [100.0] * 399)
    starts = [df.index[0], df.index[500]]
    out = pullbacks.floor_breaches(df, starts, depths=(0.40, 0.50), years=(1, 2))
    assert out[0.40][2] == 0.5 and out[0.50][2] == 0.0  # only the first start saw the 45% dip, within two years
    assert out[0.40][1] == 0.0  # day 400 is more than a year after the first start


def test_new_cash_can_carry_more_leverage_the_nearer_price_is_to_the_floor():
    caps = [pullbacks.add_leverage_cap(p, 1300.0, 0.005) for p in (1800.0, 2313.0, 4000.0)]
    assert caps == sorted(caps, reverse=True) and caps[1] == pytest.approx(2.27, abs=0.01)
    # adding at exactly that cap leaves a position already at the floor with its liquidation price unchanged
    floor, price, mmr = 1300.0, 2313.0, 0.005
    qty, equity = 1.0, 1.0 * (price - floor * (1 - mmr))  # a position whose liquidation price is the floor
    added = 0.5
    deposit = added * price / caps[1]
    liq = ((qty + added) * price - (equity + deposit)) / ((qty + added) * (1 - mmr))
    assert liq == pytest.approx(floor)


def test_timing_study_command(tmp_path, capsys, monkeypatch):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.95).to_numpy(), start="2016-05-18")
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "prices.csv", index=False)
    monkeypatch.setattr(derivs, "fetch_funding_history",
                        lambda start, end, **kw: (pd.Series(0.0001, index=pd.date_range(start, end, freq="D", tz="UTC")), 0))
    assert main(["--provider", "csv", "--csv", str(tmp_path / "prices.csv"), "timing-study", "--floor", "1300"]) == 0
    text = capsys.readouterr().out
    for heading in ("may fall on any of 4 days", "Fixed day (the window's first)", "Lowest close of the 4 days (hindsight)",
                    "traded through a floor", "(yours: 1,300", "the most leverage each add can carry"):
        assert heading in text, heading
