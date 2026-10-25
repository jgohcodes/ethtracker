import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ethtracker import config, plan, risk, snapshot
from ethtracker.cli import main
from ethtracker.config import Config
from ethtracker.journal import Journal

from tests.conftest import synthetic_candles
from tests.fakes import FakeDashboard
from tests.test_cli import cloud_env, fake_deribit, run

CFG = Config()
MMR = CFG.maintenance_margin_rate
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def liq_of(equity, qty, price):
    return (price - equity / qty) / (1 - MMR)


# ---- sizing against the floor --------------------------------------------------------------------

def test_new_cash_can_carry_the_most_leverage_the_floor_allows_and_no_more():
    floor, price = 1250.0, 2336.0
    s = plan.sizing(CFG, {"floor": floor, "weekly_add": 5000.0}, price, 0.0, 0.0, None)
    assert s["status"] == "no position" and s["cap_leverage"] == pytest.approx(2.14, abs=0.01)
    assert s["weekly_notional"] == pytest.approx(5000.0 * s["cap_leverage"])
    assert liq_of(5000.0, s["weekly_qty"], price) == pytest.approx(floor)  # exactly at the floor, not above it


def test_a_position_at_the_floor_has_no_room_without_new_cash():
    floor, price, qty = 1250.0, 2400.0, 3.0
    equity = qty * (price - floor * (1 - MMR))  # liquidation exactly at the floor
    assert plan.room(equity, qty, price, floor, MMR) == pytest.approx(0.0, abs=1e-12)
    s = plan.sizing(CFG, {"floor": floor, "weekly_add": 500.0}, price, equity, qty, liq_of(equity, qty, price))
    assert s["status"] == "at the floor"
    assert liq_of(equity + 500.0, qty + s["weekly_qty"], price) == pytest.approx(floor)


def test_a_liquidation_above_the_floor_says_how_much_to_cut_or_deposit():
    floor, price, qty, equity = 1250.0, 2400.0, 6.0, 5000.0
    liq = liq_of(equity, qty, price)
    s = plan.sizing(CFG, {"floor": floor, "weekly_add": 500.0}, price, equity, qty, liq)
    assert liq > floor and s["status"] == "above the floor" and s["weekly_qty"] == 0.0
    assert liq_of(equity, qty - s["cut_qty"], price) == pytest.approx(floor)
    assert liq_of(equity + s["margin_needed"], qty, price) == pytest.approx(floor)


def test_the_weekly_add_buys_more_the_further_price_falls_toward_the_floor():
    floor, price, qty = 1250.0, 2400.0, 3.0
    equity = qty * (price - floor * (1 - MMR))
    s = plan.sizing(CFG, {"floor": floor, "weekly_add": 500.0}, price, equity, qty, liq_of(equity, qty, price))
    adds = [rung["add_qty"] for rung in s["ladder"]]
    assert [r["depth"] for r in s["ladder"]] == list(plan.LADDER_DEPTHS)
    assert adds == sorted(adds) and adds[0] > s["weekly_qty"]
    caps = [rung["cap_leverage"] for rung in s["ladder"]]
    assert caps == sorted(caps)  # nearer the floor, new cash carries more leverage
    assert all(r["filled"] is None for r in s["ladder"])  # no history given


def test_nothing_can_be_added_at_or_under_the_floor():
    s = plan.sizing(CFG, {"floor": 1250.0, "weekly_add": 500.0}, 1240.0, 0.0, 0.0, None)
    assert s["cap_leverage"] is None and s["weekly_qty"] == 0.0


# ---- volatility and touch odds -------------------------------------------------------------------

def test_touch_probability_matches_the_reflection_principle():
    assert plan.touch_probability(100.0, 100.0, 0.8, 1.0) == 1.0
    one_sigma_down = 100.0 * math.exp(-0.8)
    assert plan.touch_probability(100.0, one_sigma_down, 0.8, 1.0) == pytest.approx(0.3173, abs=1e-4)
    month, year = (plan.touch_probability(2336.0, 1250.0, 0.7, t) for t in (30 / 365, 1.0))
    assert 0 < month < year < 1


def test_ewma_vol_recovers_a_steady_volatility():
    r = np.tile([0.02, -0.02], 200)
    close = pd.Series(1000 * np.exp(np.cumsum(r)))
    assert plan.ewma_vol(close) == pytest.approx(0.02 * math.sqrt(365), rel=0.01)
    assert plan.ewma_vol(close.iloc[:20]) is None


def test_floor_risk_reports_distance_odds_and_history():
    df = synthetic_candles(1500, seed=3)
    price = float(df["close"].iloc[-1])
    out = plan.floor_risk(df, price, price * 0.5)
    assert out["distance"] == pytest.approx(1.0)
    assert 0 < out["touch_30d"] < out["touch_1y"] < 1 and out["sigmas"] > 0
    json.dumps(snapshot._clean(out), allow_nan=False)


# ---- trend, crowding, odds -----------------------------------------------------------------------

def test_trend_labels_need_price_averages_and_momentum_to_agree():
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    up = pd.DataFrame({"close": np.linspace(1000, 3000, 300)}, index=idx)
    down = pd.DataFrame({"close": np.linspace(3000, 1000, 300)}, index=idx)
    assert plan.trend(up)["label"] == "Uptrend" and plan.trend(up)["above_200"]
    assert plan.trend(down)["label"] == "Downtrend" and not plan.trend(down)["above_50"]
    short = pd.DataFrame({"close": np.linspace(1000, 3000, 100)}, index=idx[:100])
    assert plan.trend(short)["label"] == "Mixed" and plan.trend(short)["sma200"] is None


def test_crowding_places_funding_against_its_own_year():
    assert plan.crowding(None) is None
    idx = pd.date_range("2025-10-01", periods=100, freq="D", tz="UTC")
    rising = pd.Series(np.linspace(0.00001, 0.0002, 100), index=idx)
    c = plan.crowding(rising)
    assert c["percentile"] == 1.0 and c["label"] == "Crowded longs" and c["days"] == 100
    flipped = rising.copy()
    flipped.iloc[-1] = -0.00001
    assert plan.crowding(flipped)["label"] == "Light longs"


def test_odds_lists_the_baselines_first_and_every_signal_with_its_record():
    df = synthetic_candles(1500, seed=11)
    out = plan.odds(df)
    names = [r["name"] for r in out["signals"]]
    assert names[:2] == ["Every day", "Every pullback day"] and len(names) > 2
    assert all(r["baseline"] for r in out["signals"][:2]) and not any(r["baseline"] for r in out["signals"][2:])
    assert all(r["held"] is None or 0 <= r["held"] <= 1 for r in out["signals"])
    for r in out["signals"][2:]:
        assert (r["last_fired"] is None) == (r["days_ago"] is None)


# ---- the snapshot under a plan -------------------------------------------------------------------

def plan_journal(tmp_path, price):
    j = Journal(tmp_path / "state.json")
    j.set_equity(5000.0)
    j.set_plan(price * 0.55, 500.0)
    j.open(side="long", entry=price, qty=4.0, leverage=1.9, stop=price * 0.55, target=None, mmr=MMR,
           initial_risk_usd=1.0, note="plan add")
    return j


def test_a_plan_puts_sizing_and_floor_risk_in_the_snapshot_and_drops_the_swing_rules(tmp_path):
    df = synthetic_candles(900, seed=5)
    price = float(df["close"].iloc[-1])
    snap = snapshot.build(CFG, plan_journal(tmp_path, price), df, price, now=NOW, history=df)
    json.dumps(snap, allow_nan=False)
    assert snap["plan"]["floor"] == pytest.approx(price * 0.55) and snap["plan"]["status"] in ("at the floor", "below the floor")
    assert len(snap["plan"]["ladder"]) == len(plan.LADDER_DEPTHS) and snap["plan"]["ladder"][0]["filled"] is not None
    assert snap["floor_risk"]["touch_1y"] is not None
    assert {"trend", "crowding", "odds"} <= set(snap) and snap["crowding"] is None
    (pos,) = snap["positions"]
    assert pos["max_days"] is None and snap["config"]["max_holding_days"] is None
    assert pos["action"] == "HOLD" and "adding plan" in pos["reasons"][0]


def test_without_a_plan_the_snapshot_still_carries_the_market_context(tmp_path):
    df = synthetic_candles(500)
    j = Journal(tmp_path / "state.json")
    snap = snapshot.build(CFG, j, df, float(df["close"].iloc[-1]), now=NOW)
    assert snap["plan"] is None and snap["floor_risk"] is None and snap["trend"]["label"] in ("Uptrend", "Downtrend", "Mixed")


def test_the_public_summary_never_carries_the_plan(tmp_path):
    df = synthetic_candles(900, seed=5)
    price = float(df["close"].iloc[-1])
    j = plan_journal(tmp_path, price)
    j.set_plan(1237.0, 433.0)
    line = snapshot.public_summary(snapshot.build(CFG, j, df, price, now=NOW, history=df))
    for secret in ("1237", "1,237", "433", "floor", "plan"):
        assert secret not in line


# ---- the plan and add commands -------------------------------------------------------------------

def test_the_first_plan_needs_a_floor_and_a_weekly_add(capsys):
    code = main(["plan", "--floor", "1250"])
    assert code == 2 and "both --floor and --weekly" in capsys.readouterr().err
    code, out = run(capsys, "plan", "--floor", 1250, "--weekly", 500, "--price", 2336)
    assert code == 0 and "never above $1,250.00" in out and "2.14x" in out
    code, out = run(capsys, "plan", "--weekly", 400, "--price", 2336)  # one field at a time after that
    assert code == 0 and "add $400.00 a week" in out and "$1,250.00" in out


def test_add_needs_a_plan(capsys):
    assert main(["add", "--usd", "5000", "--price", "2336"]) == 2
    assert "set a plan first" in capsys.readouterr().err


def test_add_buys_the_most_that_keeps_liquidation_at_or_under_the_floor(capsys):
    run(capsys, "plan", "--floor", 1250, "--weekly", 500, "--price", 2336)
    code, out = run(capsys, "add", "--usd", 5000, "--price", 2336)
    assert code == 0 and "$5,000.00 added" in out
    j = Journal(config.home() / "state.json")
    (pos,) = j.open_positions()
    assert j.equity == 5000.0 and pos.stop == 1250.0 and pos.note == "plan add"
    assert 4.55 < pos.qty < 4.58  # 5000 / (2336 - 1250 x 0.995), less the entry fee
    status = risk.mark_position(CFG, pos, 2336.0, datetime.now(timezone.utc))
    liq = risk.account_liquidation_price(j.equity + status.unrealized, [pos], 2336.0, MMR)
    assert liq <= 1250.0 and liq > 1240.0

    code, out = run(capsys, "add", "--price", 2200)  # the weekly add, by default
    assert code == 0 and "$500.00 added" in out
    j = Journal(config.home() / "state.json")
    assert j.equity == 5500.0 and len(j.open_positions()) == 2


def test_add_refuses_a_size_that_lifts_liquidation_above_the_floor_unless_forced(capsys):
    run(capsys, "plan", "--floor", 1250, "--weekly", 500, "--price", 2336)
    assert main(["add", "--usd", "500", "--price", "2400", "--qty", "1"]) == 2
    assert "would lift the liquidation price above $1,250.00" in capsys.readouterr().err
    assert Journal(config.home() / "state.json").equity is None  # nothing recorded
    code, _ = run(capsys, "add", "--usd", 500, "--price", 2400, "--qty", 1, "--force")
    assert code == 0 and len(Journal(config.home() / "state.json").open_positions()) == 1


def test_add_refuses_a_price_at_or_under_the_floor(capsys):
    run(capsys, "plan", "--floor", 1250, "--weekly", 500, "--price", 2336)
    assert main(["add", "--price", "1200"]) == 2
    assert "at or under the floor" in capsys.readouterr().err


def test_cloud_run_logs_never_show_the_plan(capsys, monkeypatch, candles_csv, tmp_path):
    j = Journal(tmp_path / "cloud" / "state.json")
    j.set_equity(4321.0)
    j.set_plan(1237.0, 433.0)
    j.save()
    cloud_env(monkeypatch)
    dash = FakeDashboard(state={"journal": j.path.read_text()}).install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    captured = capsys.readouterr()
    for secret in ("1237", "1,237", "433", "4321", "4,321", "floor"):
        assert secret not in captured.out + captured.err, secret
    assert dash.snapshots[0]["plan"]["floor"] == 1237.0  # it reaches the private dashboard only


def test_under_a_plan_the_weekly_buys_show_as_one_position_with_each_buy_listed(tmp_path):
    df = synthetic_candles(900, seed=5)
    price = float(df["close"].iloc[-1])
    j = plan_journal(tmp_path, price)
    for entry in (price * 0.9, price * 1.1):
        j.open(side="long", entry=entry, qty=1.0, leverage=2.0, stop=price * 0.55, target=None, mmr=MMR,
               initial_risk_usd=1.0, note="plan add")
    snap = snapshot.build(CFG, j, df, price, now=NOW, history=df)
    (book,) = snap["positions"]
    assert book["qty"] == pytest.approx(6.0) and book["entry"] == pytest.approx(price)  # (4p + 0.9p + 1.1p) / 6
    assert book["stop"] == pytest.approx(price * 0.55) and book["action"] == "HOLD"
    assert [lot["qty"] for lot in book["lots"]] == [4.0, 1.0, 1.0]
    assert sum(lot["unrealized"] for lot in book["lots"]) == pytest.approx(book["unrealized"])
    assert len(json.dumps(snap)) < 256 * 1024


def test_context_prints_the_market_panels_and_a_floor_only_when_asked(capsys, tmp_path):
    df = synthetic_candles(1500, seed=9)
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(tmp_path / "eth.csv", index=False)
    code, text = run(capsys, "--provider", "csv", "--csv", tmp_path / "eth.csv", "context")
    assert code == 0
    for heading in ("Trend:", "Crowding: funding history unavailable", "Volatility:", "Reaching a floor",
                    "Every pullback day", "A buy order under the close"):
        assert heading in text, heading
    assert "$1,237" not in text
    code, text = run(capsys, "--provider", "csv", "--csv", tmp_path / "eth.csv", "context", "--floor", 1237)
    assert code == 0 and "$1,237" in text
