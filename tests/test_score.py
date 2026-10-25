import json

import numpy as np
import pandas as pd
import pytest

from ethtracker import derivs, score, signals, snapshot
from ethtracker.config import Config

from tests.conftest import synthetic_candles
from tests.test_cli import run
from tests.test_cycles import path

CFG = Config()


def test_the_scored_states_follow_the_same_rules_as_the_dashboards_signals():
    df = synthetic_candles(400, seed=12)
    st = score.states(CFG, df)
    names = {"trend": "trend", "long_trend": "long_trend", "volume": "volume_pressure"}
    for back in range(30):
        sub = df.iloc[: len(df) - back]
        sigs = signals.compute(CFG, sub)
        for key, name in names.items():
            g = sigs.get(name)
            assert st[key].iloc[len(sub) - 1] == (g.bias if g else 0), (key, back)


def test_momentum_needs_both_horizons_to_agree():
    closes = [100.0] * 40 + list(np.linspace(101.0, 115.0, 15)) + [105.0] * 7  # up over 28 days, down over 7
    st = score.states(CFG, path(closes))
    assert st["momentum"].iloc[-1] == 0
    assert st["momentum"].iloc[-8] == 1  # at the top of the climb both windows were up
    assert np.isnan(st["momentum"].iloc[20])  # not 28 days of history yet


def test_the_score_is_the_weighted_average_of_what_can_be_read():
    st = pd.DataFrame({"momentum": [1.0, -1.0, 1.0], "trend": [1.0, -1.0, 0.0], "long_trend": [1.0, -1.0, -1.0],
                       "volume": [1.0, -1.0, 0.0], "crowding": [np.nan, -1.0, np.nan]})
    sc = score.combine(st)
    assert sc.iloc[0] == pytest.approx(1.0) and sc.iloc[1] == pytest.approx(-1.0)
    w = score.WEIGHTS
    assert sc.iloc[2] == pytest.approx((w["momentum"] - w["long_trend"]) / (1 - w["crowding"]))  # crowding missing: rescaled
    assert sum(w.values()) == pytest.approx(1.0)


def test_crowding_is_negative_only_when_funding_is_in_the_top_tenth_of_its_year():
    df = synthetic_candles(400, seed=2)
    funding = pd.Series(np.linspace(0.00001, 0.0002, 400), index=df.index)
    st = score.states(CFG, df, funding)
    assert st["crowding"].iloc[-1] == -1 and np.isnan(st["crowding"].iloc[10])
    falling = pd.Series(np.linspace(0.0002, 0.00001, 400), index=df.index)
    assert score.states(CFG, df, falling)["crowding"].iloc[-1] == 0


def test_forward_returns_and_falls_look_only_ahead():
    closes = [100.0] * 5 + [90.0] * 40
    fwd = score.forward(path(closes), horizon=30)
    assert fwd["ret"].iloc[4] == pytest.approx(-0.10) and bool(fwd["fell"].iloc[4]) is True
    assert fwd["ret"].iloc[5] == pytest.approx(0.0) and bool(fwd["fell"].iloc[5]) is False
    assert np.isnan(fwd["ret"].iloc[-1])


def test_buckets_cover_the_whole_range():
    assert [score.label(v) for v in (-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0)] == [
        "Strongly negative", "Strongly negative", "Leaning negative", "Neutral", "Leaning positive",
        "Strongly positive", "Strongly positive"]


def test_shuffled_days_carry_no_edge_on_average():
    noise = score.chance_ic(CFG, synthetic_candles(1200, seed=4), runs=8)
    assert 0 < noise < 0.3


def test_the_dashboard_block_has_strips_history_and_a_calibration():
    df = synthetic_candles(900, seed=6)
    funding = pd.Series(0.0001, index=df.index)
    b = score.block(CFG, df, funding, df, None)
    json.dumps(snapshot._clean(b), allow_nan=False)
    assert -1 <= b["value"] <= 1 and b["label"] == score.label(b["value"])
    assert len(b["history"]) == score.HISTORY_DAYS and set(b["ago"]) == {"3", "7", "14"}
    assert b["ago"]["3"] == pytest.approx(b["history"][-4]["score"], abs=1e-4)
    assert [c["key"] for c in b["components"]] == [c.key for c in score.COMPONENTS]
    assert all(len(c["strip"]) == score.STRIP_DAYS for c in b["components"])
    assert sum(c["contribution"] for c in b["components"] if c["contribution"] is not None) == pytest.approx(b["value"])
    names = {u["name"] for u in b["unscored"]}
    assert "rsi" in names and not names & score.SCORED_ELSEWHERE
    assert all(len(u["strip"]) == score.STRIP_DAYS for u in b["unscored"])
    cal = b["calibration"]
    assert cal["bucket"] == b["label"] and cal["same"]["days"] > 0 and cal["all"]["days"] >= cal["same"]["days"]


def test_the_snapshot_carries_the_score(tmp_path):
    from ethtracker.journal import Journal
    df = synthetic_candles(500)
    snap = snapshot.build(CFG, Journal(tmp_path / "s.json"), df, float(df["close"].iloc[-1]), history=df)
    assert snap["score"]["components"] and snap["score"]["label"]


def test_score_study_command(capsys, tmp_path, monkeypatch):
    df = synthetic_candles(1500, seed=9)
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    monkeypatch.setattr(derivs, "fetch_funding_history",
                        lambda start, end, **kw: (pd.Series(0.0001, index=pd.date_range(start, end, freq="D", tz="UTC")), 0))
    monkeypatch.setattr(score, "chance_ic", lambda *a, **k: 0.08)
    code, text = run(capsys, "--provider", "csv", "--csv", tmp_path / "eth.csv", "score-study")
    assert code == 0
    for heading in ("Each part against ETH's next 30 days", "Chance alone", "literature weights (the score)",
                    "What followed each score bucket", "Today: score"):
        assert heading in text, heading
