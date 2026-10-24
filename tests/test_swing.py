from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ethtracker import cycles, data, derivs, swing
from ethtracker.cli import main
from ethtracker.config import Config
from ethtracker.data import DataError
from ethtracker.store import CachedProvider

from tests.conftest import synthetic_candles
from tests.test_cycles import FREE, cycle_path, path


def test_coinbase_serves_the_product_asked_for_and_caches_it_separately(tmp_path):
    btc = data.get_provider("coinbase", product="BTC-USD")
    assert btc.product == "BTC-USD" and btc.spot_url.endswith("/BTC-USD/spot")
    assert data.get_provider("coinbase").product == "ETH-USD"
    assert CachedProvider(btc, tmp_path).path.name == "coinbase-BTC-USD-1d.csv"
    assert CachedProvider(data.get_provider("coinbase"), tmp_path).path.name == "coinbase-ETH-USD-1d.csv"  # unchanged
    with pytest.raises(DataError, match="ETH-USD only"):
        data.get_provider("kraken", product="BTC-USD")


def test_daily_funding_can_replace_the_configured_rate():
    df = synthetic_candles(n=300, seed=3)
    cfg = Config()
    constant = np.full(len(df), cfg.funding_rate_8h * 3)
    assert cycles.simulate("x", df, cfg, leverage=2.5, funding=constant).multiple == pytest.approx(
        cycles.simulate("x", df, cfg, leverage=2.5).multiple)
    assert cycles.simulate("x", df, cfg, leverage=2.5, funding=np.zeros(len(df))).multiple == pytest.approx(
        cycles.simulate("x", df, replace(cfg, funding_rate_8h=0.0), leverage=2.5).multiple)


def test_a_spike_is_a_30_percent_run_within_14_days():
    closes = [100.0] * 20 + [110, 120, 131, 135] + [100.0] * 20 + [100 + 2.0 * k for k in range(1, 16)]
    hits = np.flatnonzero(swing.spike_run(path(closes)))
    assert list(hits[:2]) == [22, 23]  # 131 and 135 are 30%+ above the 14-day low of 100
    assert not any(h > 44 for h in hits)  # +2% a day reaches +30% only after 15 days


@pytest.mark.parametrize("rule", [swing.spike_run, swing.spike_rsi, lambda df: swing.first_touch(df, 0.25)])
def test_the_new_rules_never_look_ahead(rule):
    df = synthetic_candles(n=600, seed=12)
    whole = rule(df)
    for cut in (200, 450, 599):
        assert np.array_equal(rule(df.iloc[: cut + 1]), whole[: cut + 1])


def test_a_pullback_bought_and_held_until_the_spike():
    # flat at 100, a 25% dip to 75 (both dip entries fire there), lower to 70, then a 30% run to 95
    closes = [100.0] * 35 + [90, 80, 75, 72, 70, 75, 85, 95, 95]
    df = path(closes)
    entries = swing.first_touch(df, 0.20)
    exits = swing.spike_run(df)
    result = cycles.simulate("x", df, FREE, leverage=1.0, entries=entries, exits=exits)
    assert result.trades == 1 and result.multiple == pytest.approx(95 / 80)


def test_every_combination_is_run_with_holding_first():
    df = cycle_path()
    rules = swing.Rules(df, FREE, None)
    rows = rules.run(df, df.index[0], df.index[-1], FREE)
    assert [r.name for r in rows[:2]] == ["Hold spot, 1x", "Hold perp, 2.5x"]
    assert len(rows) == 2 + len(swing.ENTRIES) * len(swing.EXITS) * len(swing.LEVERAGES)


def test_today_and_the_matching_depth():
    df = path([100.0] * 40 + [60.0])
    now = swing.today(df)
    assert now.below_high == pytest.approx(0.4) and now.below_30d_high == pytest.approx(0.4)
    assert swing.today_depth(df) == pytest.approx(0.4)
    assert swing.today_depth(path([100.0] * 40 + [95.0])) == 0.2  # never shallower than a 20% dip


@pytest.mark.parametrize("asset,instrument", [("ETH", "ETH-PERPETUAL"), ("BTC", "BTC-PERPETUAL")])
def test_swing_study_command(tmp_path, capsys, monkeypatch, asset, instrument):
    df = cycle_path()
    rng = np.random.default_rng(5)
    noisy = df["close"] * np.exp(rng.normal(0, 0.03, len(df)))
    df = path(noisy.to_numpy(), lows=(noisy * 0.97).to_numpy(), start="2016-05-18")
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "prices.csv", index=False)
    asked = []

    def fake_history(start, end, instrument="ETH-PERPETUAL", **kwargs):
        asked.append(instrument)
        days = pd.date_range(start, end, freq="D", tz="UTC")
        return pd.Series(0.00005, index=days), 0

    monkeypatch.setattr(derivs, "fetch_funding_history", fake_history)
    assert main(["--provider", "csv", "--csv", str(tmp_path / "prices.csv"), "swing-study", "--asset", asset]) == 0
    text = capsys.readouterr().out
    assert asked == [instrument]
    for heading in (f"{asset}: buy a sharp pullback", "growth-optimal", "Whole history", "Bull phases only",
                    "Current bear phase", "years forward", "20% dip -> RSI 80, 2.5x"):
        assert heading in text, heading
