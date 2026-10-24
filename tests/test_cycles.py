from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ethtracker import cycles
from ethtracker.cli import main
from ethtracker.config import Config

from tests.conftest import synthetic_candles

FREE = replace(Config(), taker_fee_rate=0.0, funding_rate_8h=0.0)


def path(closes, lows=None, start="2024-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    lows = closes if lows is None else np.asarray(lows, dtype=float)
    return pd.DataFrame(
        {"open": closes, "high": np.maximum(closes, lows), "low": lows, "close": closes, "volume": 1.0},
        index=pd.date_range(start, periods=len(closes), freq="D", tz="UTC"),
    )


def flags(n, *days):
    out = np.zeros(n, dtype=bool)
    out[list(days)] = True
    return out


# ---- the account ---------------------------------------------------------------------------------

def test_holding_spot_is_the_price_ratio():
    df = path([100, 150, 80, 300])
    assert cycles.simulate("spot", df, FREE, leverage=1.0, core=True).multiple == pytest.approx(3.0)


@pytest.mark.parametrize("leverage", [1.0, 1.5, 2.5])
def test_a_held_perp_gains_leverage_times_the_move(leverage):
    df = path([100, 120, 110, 140])
    result = cycles.simulate("perp", df, FREE, leverage=leverage)
    assert result.multiple == pytest.approx(1 + leverage * 0.4)
    assert result.trades == 1 and result.exposure == 1.0 and result.liquidated is None


def test_liquidation_at_the_days_low_ends_the_account():
    # 2.5x with 0.5% maintenance margin liquidates about 39.7% below entry
    liq = 100 * (1 - 1 / 2.5) / (1 - FREE.maintenance_margin_rate)
    survives = cycles.simulate("ok", path([100, 100, 200], lows=[100, liq + 0.5, 200]), FREE, leverage=2.5)
    assert survives.liquidated is None and survives.multiple == pytest.approx(3.5)
    dead = cycles.simulate("dead", path([100, 100, 200], lows=[100, liq - 0.5, 200]), FREE, leverage=2.5)
    assert dead.multiple == 0.0 and dead.max_drawdown == 1.0
    assert dead.liquidated == pd.Timestamp("2024-01-02", tz="UTC")


def test_funding_and_fees_are_charged():
    cfg = replace(FREE, funding_rate_8h=0.0001)
    flat = cycles.simulate("perp", path([100] * 11), cfg, leverage=1.0)
    assert flat.multiple == pytest.approx(1 - 0.0003 * 10)  # paid on each day after the opening close
    cfg = replace(FREE, taker_fee_rate=0.001)
    traded = cycles.simulate("perp", path([100, 100]), cfg, leverage=2.0)
    assert traded.multiple == pytest.approx(1 - 2 * 0.001 * 2)  # in and out on 2x notional


def test_a_tactical_trade_opens_at_the_entry_close_and_closes_at_the_exit_close():
    df = path([100, 90, 99, 120, 60])
    result = cycles.simulate("dip", df, FREE, leverage=2.0, entries=flags(5, 1), exits=flags(5, 3))
    assert result.multiple == pytest.approx(1 + 2 * (120 / 90 - 1))  # the later crash happens while flat
    assert result.trades == 1 and result.exposure == pytest.approx(2 / 5)


def test_the_time_stop_closes_a_trade_that_never_gets_overbought():
    df = path([100, 100, 110, 121, 50])
    result = cycles.simulate("dip", df, FREE, leverage=1.0, entries=flags(5, 0), max_hold=2)
    assert result.multiple == pytest.approx(1.1)


def test_the_spot_core_only_adds_leverage_on_a_signal():
    df = path([100, 100, 110, 121])
    assert cycles.simulate("core", df, FREE, leverage=2.5, core=True, entries=np.zeros(4, bool)).multiple == pytest.approx(1.21)
    result = cycles.simulate("core", df, FREE, leverage=2.5, core=True, entries=flags(4, 1), exits=flags(4, 2))
    # 1x spot throughout, plus 1.5x more from day 1's close to day 2's close
    assert result.multiple == pytest.approx(1.21 + 1.5 * 0.1)


# ---- the rules -----------------------------------------------------------------------------------

def dip_then_recovery(bottom_days=10):
    """Flat at 100, a fall to 75, a flat bottom, then two bounces above the 10-day average while
    still 20%+ below the high (78, then 79 after a dip back to 73), then a recovery to 95."""
    closes = [100.0] * 40 + list(np.linspace(97, 75, 6)) + [74.0] * bottom_days + [78.0, 73.0, 73.0, 79.0, 95.0]
    return path(closes)


def test_the_first_dip_close_fires_once_per_dip():
    df = dip_then_recovery()
    hits = np.flatnonzero(cycles.first_dip_close(df))
    assert len(hits) == 1 and df["close"].iloc[hits[0]] <= 80


def test_dip_done_waits_for_the_bounce_and_buys_once():
    df = dip_then_recovery()
    hits = np.flatnonzero(cycles.dip_done_reclaim(df))
    assert len(hits) == 1
    assert df["close"].iloc[hits[0]] == 78.0  # the first close back above the 10-day average, not the second one at 79


def test_dip_done_skips_a_dip_that_takes_too_long_to_recover():
    assert not cycles.dip_done_reclaim(dip_then_recovery(bottom_days=cycles.CONFIRM_WITHIN + 5)).any()


@pytest.mark.parametrize("rule", [cycles.first_dip_close, cycles.dip_done_reclaim, cycles.dip_done_rsi, cycles.above_trend])
def test_rules_never_look_ahead(rule):
    df = synthetic_candles(n=700, seed=11)
    whole = rule(df)
    for cut in (250, 400, 699):
        assert np.array_equal(rule(df.iloc[: cut + 1]), whole[: cut + 1])


def test_the_synthetic_history_actually_exercises_the_dip_rules():
    df = synthetic_candles(n=700, seed=11)
    assert cycles.first_dip_close(df).sum() > 2 and cycles.dip_done_reclaim(df).sum() > 1


# ---- phases --------------------------------------------------------------------------------------

def cycle_path(start="2016-05-18", end="2026-10-23"):
    """Piecewise path with known peaks and troughs. Not market data."""
    knots = {
        start: 10, "2018-01-13": 1400, "2018-12-15": 85, "2021-11-08": 4800,
        "2022-06-18": 1000, "2025-08-22": 4900, "2026-06-05": 1550, end: 2400,
    }
    days = pd.date_range(start, end, freq="D", tz="UTC")
    series = pd.Series(np.nan, index=days)
    for day, price in knots.items():
        if pd.Timestamp(day, tz="UTC") in series.index:
            series[pd.Timestamp(day, tz="UTC")] = price
    closes = np.exp(np.log(series).interpolate()).to_numpy()
    return path(closes, start=start)


def test_phases_follow_the_peaks_and_troughs():
    phases = cycles.find_phases(cycle_path())
    got = [(p.kind, f"{p.start:%Y-%m-%d}", f"{p.end:%Y-%m-%d}", p.complete, p.from_bottom) for p in phases]
    assert got == [
        ("bull", "2016-05-18", "2018-01-13", True, False),
        ("bear", "2018-01-13", "2018-12-15", True, False),
        ("bull", "2018-12-15", "2021-11-08", True, True),
        ("bear", "2021-11-08", "2022-06-18", True, False),
        ("bull", "2022-06-18", "2025-08-22", True, True),
        ("bear", "2025-08-22", "2026-10-23", False, False),
    ]
    assert phases[1].change == pytest.approx(85 / 1400 - 1)
    assert phases[-1].worst_fall == pytest.approx(1 - 1550 / 4900)


def test_phases_when_the_data_starts_inside_a_bear_market():
    phases = cycles.find_phases(cycle_path().loc["2018-08-08":])
    assert [(p.kind, f"{p.start:%Y-%m-%d}") for p in phases[:3]] == [
        ("bear", "2018-08-08"), ("bull", "2018-12-15"), ("bear", "2021-11-08")]


def test_the_last_phase_is_left_open_when_its_turning_point_cannot_be_known_yet():
    # between peak windows the next bottom is unknown, so the bear phase is still running
    phases = cycles.find_phases(cycle_path(end="2023-03-01"))
    assert (phases[-1].kind, phases[-1].complete, f"{phases[-1].start:%Y-%m-%d}") == ("bear", False, "2021-11-08")
    # inside a peak window the top is unknown, so the bull phase is still running
    phases = cycles.find_phases(cycle_path(end="2024-03-01"))
    last = phases[-1]
    assert (last.kind, last.complete, f"{last.start:%Y-%m-%d}", f"{last.end:%Y-%m-%d}") == ("bull", False, "2022-06-18", "2024-03-01")


def test_chaining_compounds_and_a_liquidation_ends_it():
    a = cycles.Result("x", 3.0, 0.2, 1, 1.0, None)
    b = cycles.Result("x", 2.0, 0.5, 2, 0.5, None)
    dead = cycles.Result("x", 0.0, 1.0, 1, 0.1, pd.Timestamp("2022-01-01", tz="UTC"))
    (both,) = cycles.chain([[a], [b]])
    assert both.multiple == pytest.approx(6.0) and both.max_drawdown == 0.5 and both.trades == 3
    (ended,) = cycles.chain([[a], [dead]])
    assert ended.multiple == 0.0 and ended.liquidated == dead.liquidated


def test_cycle_study_command(tmp_path, capsys):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.97).to_numpy(), start="2016-05-18")
    csv = tmp_path / "eth.csv"
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(csv, index=False)
    assert main(["--provider", "csv", "--csv", str(csv), "cycle-study"]) == 0
    text = capsys.readouterr().out
    for heading in ("Cycle phases", "(bottom to top)", "back to back", "Current bear phase", "Whole history", "above its 200-day average"):
        assert heading in text, heading
    assert text.count("Hold spot, 1x") == 8  # three bulls, the chain, the current bear, rolling, two whole-history tables
    assert "separate drawdowns" in text and "Today:" in text


def test_an_isolated_liquidation_costs_only_that_positions_margin():
    # 10x isolated on 25% of equity: liquidated about 9.5% below entry, losing 0.25, and the account carries on
    df = path([100, 100, 100, 150], lows=[100, 100, 89, 150])
    result = cycles.simulate("iso", df, FREE, leverage=10.0, entries=flags(4, 0, 2), isolated=0.25)
    assert result.lost_positions == 1 and result.liquidated is None and result.trades == 2
    # re-entered at day 2's close with 25% of the 0.75 left, at 10x: +50% on 1.875 notional
    assert result.multiple == pytest.approx(0.75 + 0.25 * 0.75 * 10 * 0.5)
    survives = cycles.simulate("iso", path([100, 100, 120], lows=[100, 91, 120]), FREE, leverage=10.0, isolated=0.25,
                               entries=flags(3, 0))
    assert survives.lost_positions == 0 and survives.multiple == pytest.approx(1 + 0.25 * 10 * 0.2)


def test_spot_held_against_borrowed_cash_is_liquidated_rather_than_going_negative():
    # a losing add-on leaves the spot core funded with borrowed cash; a later crash must end the account at zero
    df = path([100, 100, 70, 70, 20, 20], lows=[100, 100, 70, 70, 20, 20])
    result = cycles.simulate("core", df, FREE, leverage=2.5, core=True, entries=flags(6, 1), exits=flags(6, 2))
    assert result.liquidated == pd.Timestamp("2024-01-05", tz="UTC")
    assert result.multiple == 0.0 and result.max_drawdown == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_no_strategy_ever_reports_negative_equity(seed):
    df = synthetic_candles(n=900, seed=seed)
    df.index = pd.date_range("2019-01-01", periods=len(df), freq="D", tz="UTC")
    for result in cycles.run_window(df, df.index[0], df.index[-1], FREE):
        assert 0.0 <= result.max_drawdown <= 1.0 and result.multiple >= 0.0, result.name


def test_flash_prints_are_ignored_but_real_crashes_stay():
    df = path([300, 310, 300, 150, 140], lows=[300, 0.10, 290, 100, 135])
    cleaned, days = cycles.ignore_flash_prints(df)
    assert days == [pd.Timestamp("2024-01-02", tz="UTC")]
    assert cleaned["low"].iloc[1] == 310  # the lower of that day's open and close (equal in this helper)
    assert cleaned["low"].iloc[3] == 100  # a 50% fall in a day is a crash, not a bad print
    assert cleaned["close"].equals(df["close"])


def test_deep_starts_need_a_deep_drawdown_and_a_full_horizon():
    df = cycle_path()
    starts = cycles.deep_starts(df)
    close = df["close"]
    assert starts and all(close[d] <= 0.5 * close[:d].max() for d in starts)
    assert max(starts) <= df.index[-1] - pd.DateOffset(years=4)
    groups = cycles.episodes(starts)
    assert all((b[0] - a[-1]).days > 60 for a, b in zip(groups, groups[1:]))


def test_rolling_summarises_every_strategy_across_starts():
    df = cycle_path()
    starts = cycles.deep_starts(df)[:6]
    table = cycles.rolling(df, FREE, starts)
    assert [r.name for r in table] == [s.name for s in cycles.STRATEGIES]
    spot = table[0]
    assert spot.starts == 6 and spot.beat_spot == 0.0 and spot.liquidated == 0.0
    assert all(v == pytest.approx(1.0) for v in spot.median_vs_spot_by_episode)
