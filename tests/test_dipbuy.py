import numpy as np
import pandas as pd
import pytest

from ethtracker import derivs, dipbuy

from tests.conftest import synthetic_candles
from tests.test_cli import run as cli


def go(close, low=None, deploy=None, **kw):
    close = np.asarray(close, dtype=float)
    low = close if low is None else np.asarray(low, dtype=float)
    opts = dict(leverage=None, floor=None, mmr=0.02, fee=0.0)
    opts.update(kw)
    return dipbuy.run(close, low, np.zeros(len(close)), deploy, **opts)


def test_flat_prices_return_the_money_less_fees():
    r = go([100.0] * 28, leverage=2.5, fee=0.0005)
    assert r.multiple == pytest.approx(1 - 2.5 * 0.0005) and r.liquidations == 0 and r.idle == 0


def test_saved_cash_goes_to_work_on_the_dip_day_and_the_rest_waits():
    close = [100.0] * 20 + [80.0, 100.0]
    deploy = np.zeros(22, dtype=bool)
    deploy[20] = True
    dip, weekly = go(close, deploy=deploy, leverage=1.0), go(close, leverage=1.0)
    # deposits on days 0, 7 and 14 buy at 80; day 21's deposit is still waiting
    assert dip.multiple == pytest.approx((4 + 3 / 80 * 20) / 4) and dip.idle == pytest.approx(0.25)
    assert weekly.multiple == pytest.approx(1.0)


def test_a_fall_liquidates_the_whole_account_and_deposits_carry_on():
    close = [100.0] * 7 + [60.0] * 8
    r = go(close, leverage=3.0)
    assert r.liquidations == 1
    assert r.multiple == pytest.approx(2 / 3)  # the deposits of days 7 and 14 survive: bought at 60, flat since


def test_the_floor_plan_stops_after_its_floor_is_breached():
    close = [100.0] * 7 + [40.0] * 15
    r = go(close, floor=50.0)
    assert r.liquidations == 1 and r.multiple == pytest.approx(3 / 4)  # days 7, 14 and 21's deposits stay in cash
    assert r.idle == pytest.approx(3 / 4)


def test_dip_study_command(capsys, tmp_path, monkeypatch):
    df = synthetic_candles(2200, seed=9)
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    monkeypatch.setattr(derivs, "fetch_funding_history",
                        lambda start, end, **kw: (pd.Series(0.0001, index=pd.date_range(start, end, freq="D", tz="UTC")), 0))
    code, text = cli(capsys, "--provider", "csv", "--csv", tmp_path / "eth.csv", "dip-study")
    assert code == 0
    for heading in ("year windows from a start every 4 weeks", "spot, no leverage", "1x perp", "2.5x perp", "3x perp",
                    "floor 45% under start", "dip beat", "above 1x"):
        assert heading in text, heading


def test_spot_pays_no_funding_and_cannot_be_liquidated():
    df = synthetic_candles(500, seed=2)
    df.iloc[200:260, df.columns.get_loc("low")] *= 0.05  # a crash that would wipe out any perp
    study = dipbuy.Study(df, np.full(len(df), 0.001), mmr=0.02, fee=0.0)
    pair = study.pair(df.index[0], 1, spot=True)
    assert pair.weekly.liquidations == 0 and pair.dip.liquidations == 0
    assert study.pair(df.index[0], 1, leverage=1.0).weekly.liquidations == 1
