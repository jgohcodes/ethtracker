from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ethtracker import combo, cycles
from ethtracker.cli import main
from ethtracker.config import Config

from tests.conftest import synthetic_candles
from tests.test_cycles import FREE, cycle_path, flags, path

NONE = np.zeros(1, dtype=bool)


def run(c, df, cfg=FREE, entries=None, exits=None):
    n = len(df)
    return combo.simulate(c, df, cfg, np.zeros(n, bool) if entries is None else entries,
                          np.zeros(n, bool) if exits is None else exits)


def test_an_all_in_core_matches_the_single_account_simulator_including_costs():
    df = synthetic_candles(n=400, seed=4)
    cfg = Config()
    held = run(combo.Combo("x", 1.0, 2.5, "held"), df, cfg)
    single = cycles.simulate("x", df, cfg, leverage=2.5)
    assert held.multiple == pytest.approx(single.multiple, rel=1e-9)
    assert held.core_liquidated == single.liquidated
    spot = run(combo.Combo("x", 1.0, 1.0, "spot"), df, cfg)
    assert spot.multiple == pytest.approx(cycles.simulate("x", df, cfg, leverage=1.0, core=True).multiple, rel=1e-9)


def test_swing_profit_is_swept_into_the_core_and_bought_at_its_leverage():
    df = path([100, 100, 110, 110, 121])
    buy = run(combo.Combo("x", 0.85, 2.5, "held", 10.0), df, entries=flags(5, 1), exits=flags(5, 2))
    # sleeve: 0.15 at 10x from 100 to 110 doubles to 0.30, so 0.15 is swept; the core buys 2.5 x 0.15 more at 110
    assert buy.swept == pytest.approx(0.15) and buy.swing_trades == 1
    core_qty = 2.5 * 0.85 / 100 + 2.5 * 0.15 / 110
    assert buy.multiple == pytest.approx(0.85 + 2.5 * 0.85 * 0.1 + 0.15 + core_qty * 11 + 0.15)
    margin = run(combo.Combo("x", 0.85, 2.5, "held", 10.0, "margin"), df, entries=flags(5, 1), exits=flags(5, 2))
    assert margin.multiple == pytest.approx(0.85 + 2.5 * 0.85 * 0.21 + 0.15 + 0.15)  # same swept profit, no extra size


def test_a_swing_liquidation_never_touches_the_core():
    df = path([100, 100, 100, 100], lows=[100, 100, 89, 100])
    result = run(combo.Combo("x", 0.85, 2.5, "held", 10.0), df, entries=flags(4, 1))
    assert result.swing_liquidated == 1 and result.core_liquidated is None
    assert result.multiple == pytest.approx(0.85)  # the core is back where it started; the sleeve is gone


def test_a_core_liquidation_leaves_the_sleeve():
    df = path([100, 100, 55, 60], lows=[100, 100, 55, 60])
    result = run(combo.Combo("x", 0.85, 2.5, "held", 10.0), df)
    assert result.core_liquidated == pd.Timestamp("2024-01-03", tz="UTC")
    assert result.multiple == pytest.approx(0.15)


def test_a_kept_core_is_resized_and_pays_for_it_in_a_round_trip():
    df = path([100, 80, 80, 100])
    kept = run(combo.Combo("x", 1.0, 2.5, "kept"), df)
    # down 20% at 2.5x halves equity to 0.5 and leverage hits 4x, so it is cut back to 2.5x;
    # the 25% recovery on 2.5x then adds 0.3125: a round trip that leaves a held position flat costs the kept one 19%
    assert kept.multiple == pytest.approx(0.8125)
    assert run(combo.Combo("x", 1.0, 2.5, "held"), df).multiple == pytest.approx(1.0)


def test_spot_core_buys_swept_profit_as_spot():
    df = path([100, 100, 110, 110, 121])
    result = run(combo.Combo("x", 0.85, 1.0, "spot", 10.0), df, entries=flags(5, 1), exits=flags(5, 2))
    assert result.multiple == pytest.approx(0.85 * 1.21 + 0.15 * 1.1 + 0.15)


def test_the_growth_arithmetic():
    g = combo.Growth("x", drift=0.5, vol=0.85)
    assert g.at(1.0, 0.0) == pytest.approx(0.5)  # holding spot grows at the drift
    k = g.kelly(0.11)
    assert g.at(k, 0.11) > g.at(k * 0.8, 0.11) and g.at(k, 0.11) > g.at(k * 1.2, 0.11)
    assert g.at(2 * k, 0.11) == pytest.approx(0.0, abs=1e-12)
    df = synthetic_candles(n=500, seed=2)
    measured = combo.growth(df, "x")
    returns = np.log(df["close"]).diff().dropna()
    assert measured.drift == pytest.approx(returns.mean() * 365) and measured.vol == pytest.approx(returns.std() * np.sqrt(365))


def test_rolling_accepts_the_two_sleeve_runner():
    df = cycle_path()
    starts = cycles.deep_starts(df)[:4]
    runner = lambda f, s, e, c, r: combo.run_window(f, s, e, c, r)
    table = cycles.rolling(df, FREE, starts, runner=runner)
    assert [r.name for r in table] == [c.name for c in combo.COMBOS]
    assert table[0].beat_spot == 0.0 and table[0].liquidated == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_no_combination_ever_goes_below_zero(seed):
    df = synthetic_candles(n=900, seed=seed)
    for result in combo.run_window(df, df.index[0], df.index[-1], Config()):
        assert result.multiple >= -1e-9 and 0.0 <= result.max_drawdown <= 1.0 + 1e-9, result.name


def test_combo_study_command(tmp_path, capsys):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.97).to_numpy(), start="2016-05-18")
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    assert main(["--provider", "csv", "--csv", str(tmp_path / "eth.csv"), "combo-study"]) == 0
    text = capsys.readouterr().out
    for heading in ("arithmetic of leverage", "growth-optimal", "back to back", "Current bear phase", "separate drawdowns",
                    "Funding sensitivity", "Whole history"):
        assert heading in text, heading
