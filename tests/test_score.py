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
    assert cal["recent_from"] == score.RECENT and cal["all_recent"]["days"] <= cal["all"]["days"]


def test_crowding_failed_the_check_and_carries_no_weight():
    crowding = next(c for c in score.COMPONENTS if c.key == "crowding")
    assert crowding.weight == 0 and crowding.evidence == "failed"


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

    class FG:
        def json(self):
            return {"data": [{"timestamp": str(int(t.timestamp())), "value": str(10 + i % 85)} for i, t in enumerate(df.index)]}
    monkeypatch.setattr("ethtracker.cli.requests.get", lambda *a, **k: FG())
    code, text = run(capsys, "--provider", "csv", "--csv", tmp_path / "eth.csv", "score-study")
    assert code == 0
    for heading in ("Each part against ETH's next 30 days", "Chance alone", "literature weights (the score)",
                    "What followed each score bucket", "Today: score", "Entry setups", "50d rising, RSI <= 40",
                    "Bullish RSI divergence", "Fear & Greed <= 25 (extreme fear)", "Fear & Greed (raw)"):
        assert heading in text, heading


def test_turning_points_need_a_full_swing_and_alternate():
    closes = [100.0, 120.0, 140.0, 130.0, 95.0, 90.0, 100.0, 125.0, 130.0, 85.0]
    pts = score.pivots(path(closes)["close"], swing=0.30)
    assert [(k, p) for _, k, p in pts] == [("top", 140.0), ("bottom", 90.0), ("top", 130.0)]


def test_a_score_that_turns_after_the_top_has_a_positive_lag_and_one_before_a_negative_one():
    closes = [100.0, 120.0, 140.0, 130.0, 95.0, 90.0, 80.0, 70.0]
    df = path(closes)
    late = pd.Series([0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5], index=df.index)
    (top,) = [r for r in score.turns(df["close"], late) if r["kind"] == "top"]
    assert top["lag"] == 2 and top["at_flip"] == pytest.approx(95 / 140 - 1)
    assert top["left"] == pytest.approx((70 - 95) / (70 - 140))  # the swing runs to the last close: still running
    early = pd.Series([0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5], index=df.index)
    (top,) = [r for r in score.turns(df["close"], early) if r["kind"] == "top"]
    assert top["lag"] == -1


def test_a_score_made_of_the_future_leads_and_one_made_of_the_past_echoes():
    df = synthetic_candles(800, seed=3)
    close = df["close"]
    future = np.sign(close.shift(-7) / close - 1)
    assert score.lead(close, future, horizons=(7,))[7] > 0.8
    past = np.sign(close / close.shift(7) - 1)
    assert score.echo(close, past, horizons=(7,))[7] > 0.8
    assert score.stale(close, future, ages=(0, 60), horizon=7)[60] < 0.3
    assert score.flips_per_year(pd.Series([1.0, 1.0, -1.0, -1.0, 1.0], index=pd.date_range("2024-01-01", periods=5, freq="182D"))) > 0


def test_a_swing_the_score_never_turned_for_is_marked_not_given_a_lag():
    closes = [100.0, 140.0, 95.0, 90.0, 125.0, 80.0, 70.0]  # top, bottom, a bounce to 125, then lower
    df = path(closes)
    bearish = pd.Series([0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5], index=df.index)  # never positive on the bounce
    rows = score.turns(df["close"], bearish)
    bounce = [r for r in rows if r["kind"] == "bottom"][0]
    assert bounce["flip"] is None and bounce["lag"] is None


def test_each_scored_line_sits_on_the_side_of_its_thresholds_that_its_state_says():
    df = synthetic_candles(700, seed=21)
    funding = pd.Series(np.random.default_rng(3).normal(0.0001, 0.0001, len(df)), index=df.index)
    st, rd = score.states(CFG, df, funding), score.readings(df, funding)
    lo, hi = score.LINES["volume"][3]
    checks = {
        "momentum": (lambda d: (rd["momentum"] > 0) & (rd["momentum_2"] > 0), lambda d: (rd["momentum"] < 0) & (rd["momentum_2"] < 0)),
        "trend": (lambda d: (rd["trend"] > 0) & (rd["trend_2"] > 0), lambda d: (rd["trend"] < 0) & (rd["trend_2"] < 0)),
        "long_trend": (lambda d: rd["long_trend"] > 0, lambda d: rd["long_trend"] < 0),
        "volume": (lambda d: rd["volume"] >= hi - 1e-9, lambda d: rd["volume"] <= lo + 1e-9),
        "crowding": (None, lambda d: rd["crowding"] >= score.CROWDED),
    }
    for key, (pos, neg) in checks.items():
        if pos is not None:
            assert (st[key] == 1).any() and pos(None)[st[key] == 1].all(), key
        assert (st[key] == -1).any() and neg(None)[st[key] == -1].all(), key
        assert rd[key].notna().sum() >= st[key].notna().sum() - 1, key  # a line wherever there is a state


def test_the_block_draws_a_line_for_every_signal_with_its_thresholds():
    df = synthetic_candles(900, seed=6)
    funding = pd.Series(0.0001, index=df.index)
    b = score.block(CFG, df, funding, df, None)
    today = score.readings(df, funding).iloc[-1]
    for c in b["components"]:
        line = c["line"]
        assert len(line["values"]) == score.STRIP_DAYS and line["thresholds"] and line["unit"]
        assert line["values"][-1] == pytest.approx(today[c["key"]], rel=1e-5), c["key"]
        assert (line["second"] is not None) == (c["key"] in ("momentum", "trend"))
    rsi = next(u for u in b["unscored"] if u["name"] == "rsi")
    assert rsi["line"]["thresholds"] == [CFG.rsi_oversold, CFG.rsi_overbought] and len(rsi["line"]["risks"]) == score.STRIP_DAYS
    assert f"RSI {rsi['line']['values'][-1]:.0f}" in rsi["headline"]
    vol = next(u for u in b["unscored"] if u["name"] == "volatility")
    assert all(s == 0 for s in vol["strip"] if s is not None) and vol["line"]["values"][-1] > 0


def test_a_trade_sees_the_score_and_signals_of_the_close_before_its_day_and_nothing_after():
    df = synthetic_candles(600, seed=8)
    day = df.index[400]
    at = (day + pd.Timedelta(hours=17)).isoformat()
    (then, missing) = score.at_times(CFG, df, None, None, [at, None])
    assert missing is None and then["basis_close"] == f"{df.index[399]:%Y-%m-%d}"
    assert then["score"] == pytest.approx(score.combine(score.continuous(df)).iloc[399])
    assert then["label"] == score.label(then["score"])
    assert [c["key"] for c in then["components"]] == [c.key for c in score.COMPONENTS]
    assert not {g["name"] for g in then["firing"]} & score.SCORED_ELSEWHERE
    assert all(g["bias"] or g["risk"] for g in then["firing"])
    later = df.copy()
    later.iloc[400:, later.columns.get_loc("close")] *= 3  # the trade's own day and after change nothing
    (again,) = score.at_times(CFG, later, None, None, [at])
    assert again["score"] == pytest.approx(then["score"]) and again["firing"] == then["firing"]


def test_the_snapshot_gives_each_open_exchange_trade_its_signals_and_the_position_a_weighted_entry_score(tmp_path):
    from ethtracker.journal import Journal
    df = synthetic_candles(500, seed=3)
    times = [f"{df.index[i]:%Y-%m-%d}T12:00:00Z" for i in (300, 450)]
    account = {"source": "coinbase", "positions": [{"product": "ETH-PERP-INTX", "side": "long", "size": 3.0, "trades": [
        {"time": times[0], "price": 100.0, "size": 1.0}, {"time": times[1], "price": 110.0, "size": 2.0},
        {"time": None, "price": None, "size": 0.5}]}], "summary": None}
    snap = snapshot.build(CFG, Journal(tmp_path / "s.json"), df, float(df["close"].iloc[-1]), history=df, account=account)
    (p,) = snap["account"]["positions"]
    a, b, earlier = p["trades"]
    assert earlier["signals"] is None and a["signals"]["basis_close"] == f"{df.index[299]:%Y-%m-%d}"
    assert p["entry_score"] == pytest.approx((a["signals"]["score"] * 1 + b["signals"]["score"] * 2) / 3)
    assert "signals" not in account["positions"][0]["trades"][0]  # the input is left as it was
    assert {"date", "score", "close"} <= set(snap["score"]["history"][-1]) and snap["score"]["history"][-1]["close"] == pytest.approx(float(df["close"].iloc[-1]), abs=0.01)


def test_the_continuous_parts_keep_the_steps_signs_and_move_inside_a_state():
    df = synthetic_candles(700, seed=21)
    st, ct = score.states(CFG, df), score.continuous(df)
    for key in ("momentum", "trend", "long_trend", "volume"):
        both = st[key].notna() & ct[key].notna()
        assert (ct.loc[both & (st[key] == 1), key] > 0).all(), key
        assert (ct.loc[both & (st[key] == -1), key] < 0).all(), key
        assert ct[key].dropna().between(-1, 1).all(), key
    fine, step = score.combine(ct), score.combine(st)
    held = (step.diff() == 0) & step.notna()
    assert (fine.diff()[held].abs() > 1e-9).mean() > 0.9  # it moves on days the step score does not


def test_the_block_shows_the_continuous_score_and_keeps_the_step_one():
    df = synthetic_candles(900, seed=6)
    b = score.block(CFG, df, pd.Series(0.0001, index=df.index), df, None)
    funding = pd.Series(0.0001, index=df.index)
    assert b["value"] == pytest.approx(score.combine(score.continuous(df, funding)).iloc[-1])
    assert b["step"] == pytest.approx(score.combine(score.states(CFG, df, funding)).iloc[-1])


def test_trading_the_crossings_buys_on_the_way_up_and_sells_on_the_way_down():
    idx = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    close = pd.Series([100.0, 100.0, 110.0, 120.0, 90.0, 80.0], index=idx)
    sc = pd.Series([-0.5, 0.3, 0.4, -0.2, -0.3, -0.4], index=idx)
    r = score.cross_strategy(close, sc, fee=0.0)
    assert r.buys == 1 and r.multiple == pytest.approx(1.2)  # bought at 100, sold at 120
    half = score.cross_strategy(close, sc, sell=0.5, fee=0.0)
    assert half.multiple == pytest.approx(0.6 + 0.5 * 0.8)  # half sold at 120, half still held at 80
    tp = score.cross_strategy(close, sc, take_profit=0.1, fee=0.0)
    assert tp.multiple == pytest.approx(1.1)  # out at 110
    assert score.hold(close).multiple == pytest.approx(0.8)


def test_the_trading_signal_follows_the_step_scores_crossings():
    idx = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    step = pd.Series([-0.3, 0.0, 0.2, 0.0, -0.1, -0.5], index=idx)
    close = pd.Series([1.0, 2, 3, 4, 5, 6], index=idx)
    sig = score.trading_signal(step, close, 4)
    assert sig["state"] == "out" and sig["since"] == "2026-01-05" and sig["step"] == -0.5
    assert [(f["date"], f["to"]) for f in sig["flips"]] == [("2026-01-03", "in"), ("2026-01-05", "out")]
