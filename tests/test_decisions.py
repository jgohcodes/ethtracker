import json

import pandas as pd
import pytest

from ethtracker import config, decisions, snapshot
from ethtracker.cli import main
from ethtracker.config import Config
from ethtracker.journal import Journal

from tests.conftest import synthetic_candles
from tests.test_cli import run
from tests.test_cycles import path

CFG = Config()


def test_a_buy_is_judged_on_the_close_before_it_with_every_part_of_the_score_kept():
    df = synthetic_candles(600, seed=3)
    at = df.index[400] + pd.Timedelta(hours=14)
    record = decisions.capture(CFG, df, at, 2000.0, floor=1000.0)
    assert record["basis_close"] == f"{df.index[399]:%Y-%m-%d}"  # the buy's own day is not complete yet
    assert record["score"] == sum(record["parts"].values())
    assert record["stance"] == {1: "with", -1: "against", 0: "mixed"}[max(-1, min(1, record["score"]))]
    assert record["signals"] and {"name", "bias", "risk", "headline"} <= set(record["signals"][0])
    assert record["at"].startswith(f"{df.index[400]:%Y-%m-%d}T14:00") and record["v"] == decisions.SCHEMA
    assert 0 < record["floor"]["touch_1y"] < 1
    json.dumps(record, allow_nan=False)
    # nothing after the buy changes what was captured
    later = df.copy()
    later.iloc[400:, :4] *= 3
    assert decisions.capture(CFG, later, at, 2000.0)["readings"] == decisions.capture(CFG, df, at, 2000.0)["readings"]


def test_an_outcome_reports_each_horizon_the_30_day_low_and_who_was_right():
    closes = [100.0] * 50 + [95.0] * 10 + [130.0] * 400
    lows = list(closes)
    lows[52] = 85.0  # a 15% fall two days after the buy
    df = path(closes, lows=lows)
    record = {"at": f"{df.index[50]:%Y-%m-%d}T12:00:00+00:00", "price": 100.0, "score": -2, "risk_flags": 0,
              "signals": [{"name": "trend", "bias": -1, "headline": "Downtrend"}, {"name": "rsi", "bias": 1, "headline": "RSI low"}]}
    o = decisions.outcome(df, record)
    assert o["returns"][7] == pytest.approx(-0.05) and o["returns"][30] == pytest.approx(0.30) and o["returns"][365] == pytest.approx(0.30)
    assert o["low_30"] == pytest.approx(-0.15) and o["fell_10"] is True and o["best_close_30"] == pytest.approx(-0.05)
    assert o["judged_on"] == 90 and o["right"] is True and o["signals_right"] is False  # went against them, and was right
    assert [m["name"] for m in o["misses"]] == ["trend"]  # said down; price was up 30% a month on
    assert any("while the signals were against" in u for u in o["unflagged"])


def test_a_recent_buy_is_not_judged_yet():
    df = synthetic_candles(300)
    record = {"at": f"{df.index[-5]:%Y-%m-%d}T10:00:00+00:00", "price": float(df["close"].iloc[-5]), "score": 1, "signals": []}
    o = decisions.outcome(df, record)
    assert o["returns"][30] is None and o["right"] is None and o["judged_on"] is None and o["misses"] == []


def test_review_counts_going_against_the_signals_and_each_signals_hit_rate():
    df = synthetic_candles(800, seed=8)
    rows = []
    for k, i in enumerate((300, 400, 500)):
        at = df.index[i] + pd.Timedelta(hours=9)
        rows.append({"position_id": k + 1, "usd": 500.0, "qty": 0.2, **decisions.capture(CFG, df, at, float(df["close"].iloc[i]))})
    block = decisions.review(df, rows + [{"position_id": 9, "at": "x", "price": 1.0, "error": "no data"}], {})
    assert [r["position_id"] for r in block["rows"]] == [3, 2, 1] and block["missing"] == 1
    s = block["summary"]
    assert s["buys"] == 3 and sum(g["buys"] for g in s["by_stance"].values()) == 3
    assert all(0 <= h["right"] <= 1 for h in s["signal_hits"])


def test_add_stores_the_signals_with_the_buy_and_review_reads_them(capsys, tmp_path):
    df = synthetic_candles(900, seed=5)
    df.index = df.index + (pd.Timestamp.now(tz="UTC").normalize() - df.index[-1])
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    csv = ["--provider", "csv", "--csv", tmp_path / "eth.csv"]
    price = float(df["close"].iloc[-60])
    run(capsys, *csv, "plan", "--floor", round(price * 0.5), "--weekly", 500, "--price", price)
    at = f"{df.index[-60]:%Y-%m-%d}T15:00"
    code, text = run(capsys, *csv, "add", "--usd", 5000, "--price", price, "--at", at, "--note", "bought the dip anyway")
    assert code == 0 and "Signals on the" in text and "Stored with the buy, timestamped" in text
    j = Journal(config.home() / "state.json")
    (pos,) = j.open_positions()
    (d,) = j.decisions
    assert pos.opened_at.startswith(f"{df.index[-60]:%Y-%m-%d}T15:00") and d["position_id"] == pos.id and d["usd"] == 5000
    assert d["basis_close"] == f"{df.index[-61]:%Y-%m-%d}"
    code, text = run(capsys, *csv, "review")
    assert code == 0 and "Your buys against the signals" in text and f"{df.index[-60]:%Y-%m-%d}" in text
    snap = snapshot.build(CFG, j, df, price, history=df)
    assert snap["buys"]["rows"][0]["note"] == "bought the dip anyway"


def test_add_refuses_a_backdated_buy_without_its_price_or_one_in_the_future(capsys):
    run(capsys, "plan", "--floor", 1250, "--weekly", 500, "--price", 2336)
    assert main(["add", "--at", "2026-01-05"]) == 2
    assert "--at needs --price" in capsys.readouterr().err
    assert main(["add", "--at", "2999-01-01", "--price", "2000"]) == 2
    assert "in the future" in capsys.readouterr().err
