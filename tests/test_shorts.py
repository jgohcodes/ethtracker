import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ethtracker import combo, config, derivs, shorts
from ethtracker.cli import main
from ethtracker.config import Config
from ethtracker.data import DataError
from ethtracker.derivs import fetch_funding_history  # the real one: conftest stubs the module attribute

from tests.conftest import synthetic_candles
from tests.test_cycles import FREE, cycle_path, flags, path


def bars(closes, highs=None, start="2024-01-01"):
    df = path(closes, start=start)
    if highs is not None:
        df["high"] = np.maximum(np.asarray(highs, dtype=float), df["close"])
    return df


# ---- Deribit funding history ---------------------------------------------------------------------

def test_funding_history_is_paged_and_averaged_by_day(monkeypatch):
    day = 86_400_000
    base = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    calls = []

    def fake(url, params):
        calls.append(params)
        start = params["start_timestamp"]
        hours = [h for h in range(start, params["end_timestamp"], 8 * 3_600_000)]
        return {"result": [{"timestamp": h, "interest_8h": 0.0001 if (h - base) // day % 2 == 0 else 0.0003} for h in hours]}

    monkeypatch.setattr(derivs, "_get_json", fake)
    monkeypatch.setattr(derivs.time, "sleep", lambda s: None)
    series, failed = fetch_funding_history(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC"))
    assert failed == 0 and len(calls) == 2  # 60 days in 30-day pages
    assert series.index[0] == pd.Timestamp("2024-01-01", tz="UTC") and len(series) == 60
    assert series.iloc[0] == pytest.approx(0.0001) and series.iloc[1] == pytest.approx(0.0003)


def test_funding_history_survives_some_failed_pages_but_not_all(monkeypatch):
    monkeypatch.setattr(derivs.time, "sleep", lambda s: None)
    pages = iter([DataError("boom"), {"result": [{"timestamp": 1_704_067_200_000, "interest_8h": 0.0002}]}])

    def flaky(url, params):
        page = next(pages)
        if isinstance(page, Exception):
            raise page
        return page

    monkeypatch.setattr(derivs, "_get_json", flaky)
    series, failed = fetch_funding_history(pd.Timestamp("2023-12-01", tz="UTC"), pd.Timestamp("2024-01-30", tz="UTC"))
    assert failed == 1 and series.iloc[0] == pytest.approx(0.0002)
    monkeypatch.setattr(derivs, "_get_json", lambda url, params: {"result": []})
    with pytest.raises(DataError, match="no funding history"):
        fetch_funding_history(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-10", tz="UTC"))


def test_a_malformed_funding_payload_is_a_data_error():
    with pytest.raises(DataError):
        derivs.parse_deribit_funding_history({"result": [{"timestamp": 1}]})


# ---- signals and events --------------------------------------------------------------------------

def test_signals_fire_on_the_first_day_of_each_episode():
    df = synthetic_candles(n=600, seed=9)
    funding = pd.Series(0.0001, index=df.index)
    funding.iloc[100:105] = 0.0006
    funding.iloc[300] = 0.0009
    out = shorts.overheated(df, Config(), funding)
    hot = out["Funding at 0.05%/8h+"]
    assert list(np.flatnonzero(hot)) == [100, 300]
    for e in out.values():
        assert not (e[1:] & e[:-1]).any()  # never two days running
    assert "Funding at 0.05%/8h+" not in shorts.overheated(df, Config(), None)


def test_events_measure_the_fall_and_the_squeeze():
    closes = [100.0] * 20
    highs = [100.0] * 20
    lows = [100.0] * 20
    lows[3] = 89.0  # after a signal on day 0: a 10%+ fall
    highs[12] = 107.0  # and a spike that would liquidate a 15x short but not a 10x one
    df = path(closes, lows=lows)
    df["high"] = highs
    ev = shorts.events("x", df, FREE, flags(20, 0))
    assert ev.count == 1 and ev.median_move == 0.0 and ev.fell_10 == 1.0
    assert ev.squeeze[15.0] == 1.0 and ev.squeeze[10.0] == 0.0


# ---- short trades --------------------------------------------------------------------------------

def test_a_short_is_covered_when_it_cools_with_funding_received():
    df = bars([100, 100, 95, 90, 90])
    cool = flags(5, 3)
    flat = np.zeros(5)
    assert shorts.short_returns(df, FREE, flags(5, 1), cool, flat, 1.0) == [pytest.approx(0.10)]
    assert shorts.short_returns(df, FREE, flags(5, 1), cool, flat, 10.0) == [pytest.approx(1.0)]
    paid = np.full(5, 0.001)  # longs pay shorts 0.1% of notional a day
    got = shorts.short_returns(df, FREE, flags(5, 1), cool, paid, 1.0)[0]
    assert got == pytest.approx(0.10 + 0.001 * (95 + 90) / 100)


def test_a_squeeze_liquidates_the_short_and_the_restaked_sleeve():
    df = bars([100, 100, 100, 90], highs=[100, 100, 110, 90])
    returns = shorts.short_returns(df, FREE, flags(4, 1), flags(4, 3), np.zeros(4), 10.0)
    assert returns == [-1.0]
    result = shorts.ShortResult.of("x", 10.0, returns + [0.5])
    assert result.liquidated == 0.5 and result.restaked == 0.0
    assert shorts.short_returns(df, FREE, flags(4, 1), flags(4, 3), np.zeros(4), 1.0) == [pytest.approx(0.10)]


def test_the_time_stop_covers_a_short_that_never_cools():
    df = bars([100] + [100 + k for k in range(1, 20)])
    returns = shorts.short_returns(df, FREE, flags(20, 0), np.zeros(20, bool), np.zeros(20), 1.0, horizon=5)
    assert returns == [pytest.approx(-0.05)]


# ---- the overlay on the core ---------------------------------------------------------------------

def test_the_core_alone_matches_the_two_sleeve_simulator():
    df = synthetic_candles(n=400, seed=6)
    cfg = Config()
    funding = np.full(len(df), cfg.funding_rate_8h * 3)
    alone = shorts.core_with_derisk("x", df, cfg, funding)
    held = combo.simulate(combo.Combo("x", 1.0, 2.5, "held"), df, cfg, np.zeros(len(df), bool), np.zeros(len(df), bool))
    assert alone.multiple == pytest.approx(held.multiple, rel=1e-9)


def test_cutting_the_core_and_buying_back_the_same_quantity():
    df = path([100, 100, 90, 90, 99])
    # 2.5x held: equity 1 at 100 on day 1, cut to 1x (0.01 ETH) from 0.025 ETH
    result = shorts.core_with_derisk("x", df, FREE, np.zeros(5), entries=flags(5, 1), cool=flags(5, 3))
    # day 2: 1 - 0.01*10 = 0.90; day 3 flat, cools, buys back to 0.025; day 4: +0.025*9 = 0.225
    assert result.multiple == pytest.approx(0.90 + 0.225) and result.trades == 1
    never = shorts.core_with_derisk("x", df, FREE, np.zeros(5))
    assert never.multiple == pytest.approx(1 - 0.025 * 1)  # 1 + 0.025 * (99 - 100)


def test_a_cut_can_keep_the_core_alive():
    df = path([100, 100, 70, 100], lows=[100, 100, 59, 100])
    assert shorts.core_with_derisk("x", df, FREE, np.zeros(4)).liquidated is not None
    cut = shorts.core_with_derisk("x", df, FREE, np.zeros(4), entries=flags(4, 1), cool=np.zeros(4, bool))
    assert cut.liquidated is None


def test_short_study_command_with_and_without_funding(tmp_path, capsys, monkeypatch):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.97).to_numpy(), start="2016-05-18")
    df["high"] = noisy * 1.03
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    args = ["--provider", "csv", "--csv", str(tmp_path / "eth.csv"), "short-study"]

    def fake_history(start, end, **kwargs):
        days = pd.date_range(start, end, freq="D", tz="UTC")
        return pd.Series(np.where(np.arange(len(days)) % 50 == 0, 0.0007, 0.0001), index=days), 0

    monkeypatch.setattr(derivs, "fetch_funding_history", fake_history)
    assert main(args) == 0
    text = capsys.readouterr().out
    for heading in ("What followed each signal", "Funding at 0.05%/8h+", "Short trades", "Your 2.5x core", "beat core"):
        assert heading in text, heading

    def broken(start, end, **kwargs):
        raise DataError("blocked")

    shutil.rmtree(config.home() / "funding")  # today's cached history would otherwise be reused

    monkeypatch.setattr(derivs, "fetch_funding_history", broken)
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "Funding history unavailable (blocked)" in text and "Funding at 0.05%/8h+" not in text
