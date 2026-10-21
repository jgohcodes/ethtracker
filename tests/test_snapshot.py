import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest
import requests

from ethtracker import config, data, derivs, snapshot
from ethtracker.advice import ACTIONS
from ethtracker.config import Config
from ethtracker.journal import Journal
from tests.conftest import synthetic_candles

CFG = Config()
NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def journal_with(tmp_path, *, equity=100_000.0, entries=(250.0,)) -> Journal:
    j = Journal(tmp_path / "state.json")
    if equity is not None:
        j.set_equity(equity)
    for entry in entries:
        j.open(side="long", entry=entry, qty=10.0, leverage=5.0, stop=entry * 0.96, target=None,
               mmr=0.005, initial_risk_usd=100.0, note="swing <b>one</b>")
    return j


def build(tmp_path, cfg=CFG, **kwargs):
    candles = synthetic_candles(500)
    return snapshot.build(cfg, journal_with(tmp_path, **kwargs), candles, float(candles["close"].iloc[-1]), now=NOW, source="csv")


def test_the_snapshot_is_strict_json_with_the_fields_the_dashboard_reads(tmp_path):
    snap = build(tmp_path)
    json.dumps(snap, allow_nan=False)  # raises on NaN or infinity
    assert snap["version"] == snapshot.VERSION and snap["source"] == "csv"
    assert snap["generated_at"] == "2026-10-21T12:00:00+00:00"
    assert {"price", "price_1d", "equity", "portfolio", "positions", "closed", "signals", "votes", "levels", "notes", "config"} <= set(snap)
    assert snap["equity"]["wallet"] == 100_000.0 and snap["equity"]["total"] == pytest.approx(100_000.0 + snap["equity"]["unrealized"])
    assert snap["config"]["instrument"] == "ETH_USDC-PERPETUAL" and snap["config"]["margin_mode"] == "cross"
    assert {"sma_20", "sma_200", "avwap_low", "poc"} <= set(snap["levels"])


def test_each_position_carries_its_action_reasons_and_both_cases(tmp_path):
    (pos,) = build(tmp_path)["positions"]
    assert pos["id"] == 1 and pos["side"] == "long" and pos["action"] in ACTIONS
    assert pos["reasons"] and isinstance(pos["bull"], list) and isinstance(pos["bear"], list)
    assert pos["max_days"] == 14 and 0 <= pos["suggested_fraction"] <= 1
    assert pos["note"] == "swing <b>one</b>"  # passed through untouched: escaping is the page's job
    assert pos["liq_price"] is None and pos["dist_to_liq"] is None  # cross margin has no per-position liquidation


def test_isolated_margin_adds_the_per_position_liquidation(tmp_path):
    (pos,) = build(tmp_path, cfg=replace(CFG, margin_mode="isolated"))["positions"]
    assert pos["liq_price"] is not None and pos["dist_to_liq"] is not None


def test_the_portfolio_block_has_stress_and_whole_book_liquidation(tmp_path):
    port = build(tmp_path)["portfolio"]
    assert port["stress"]["drop_pct"] == 40.0 and isinstance(port["stress"]["survives"], bool)
    assert port["account_liq_price"] is not None and port["effective_leverage"] > 0
    assert isinstance(port["flags"], list)


def test_a_disabled_stress_rule_publishes_null_headroom_not_infinity(tmp_path):
    port = build(tmp_path, cfg=replace(CFG, stress_drop_pct=0))["portfolio"]
    assert port["stress"]["headroom_notional"] is None
    json.dumps(port, allow_nan=False)


def test_no_positions_and_no_equity_is_a_valid_empty_snapshot(tmp_path):
    snap = build(tmp_path, equity=None, entries=())
    assert snap["positions"] == [] and snap["equity"] is None and snap["portfolio"] is None
    assert snap["signals"]  # the market view still works
    only_equity = build(tmp_path / "b", equity=5_000.0, entries=())
    assert only_equity["equity"] == {"wallet": 5_000.0, "unrealized": 0.0, "total": 5_000.0, "peak": 5_000.0}
    assert only_equity["portfolio"] is None


def test_signals_are_published_with_family_votes_and_unvalidated_flags(tmp_path):
    derivs_frame = derivs.load_csv(_derivs_csv(tmp_path))
    candles = synthetic_candles(500)
    snap = snapshot.build(CFG, journal_with(tmp_path), candles, float(candles["close"].iloc[-1]), derivs_frame, now=NOW)
    by_name = {s["name"]: s for s in snap["signals"]}
    assert by_name["funding"]["unvalidated"] is True and by_name["funding"]["family"] == "derivatives"
    assert by_name["trend"]["family"] == "levels" and by_name["volatility"]["family"] == "volatility"
    assert snap["votes"]["levels"] in (-1, 0, 1)


def _derivs_csv(tmp_path):
    path = tmp_path / "derivs.csv"
    path.write_text("date,funding_8h\n" + "\n".join(f"2026-10-{d:02d},0.0007" for d in range(1, 21)) + "\n")
    return path


def test_recent_closed_trades_are_newest_first_and_capped(tmp_path):
    j = journal_with(tmp_path, entries=())
    for i in range(7):
        pos = j.open(side="long", entry=100.0 + i, qty=1.0, leverage=2.0, stop=90.0, target=None, mmr=0.005, initial_risk_usd=10.0)
        j.close(pos.id, 110.0, 5.0 if i % 2 == 0 else -10.0)
    candles = synthetic_candles(500)
    snap = snapshot.build(CFG, j, candles, 100.0, now=NOW)
    assert [c["id"] for c in snap["closed"]] == [7, 6, 5, 4, 3]
    assert snap["closed"][0]["r"] == pytest.approx(0.5) and snap["closed"][1]["r"] == pytest.approx(-1.0)


def test_clean_maps_nan_and_infinity_to_null_recursively():
    cleaned = snapshot._clean({"a": float("nan"), "b": [1.0, float("inf"), (float("-inf"), 2)], "c": {"d": np.float64("nan"), "e": 3}})
    assert cleaned == {"a": None, "b": [1.0, None, [None, 2]], "c": {"d": None, "e": 3}}
    json.dumps(cleaned, allow_nan=False)  # would raise had any NaN or infinity survived


@pytest.mark.parametrize("base,expected", [
    ("https://dash.example.app", "https://dash.example.app/api/ingest"),
    ("https://dash.example.app/", "https://dash.example.app/api/ingest"),
    ("dash.example.app", "https://dash.example.app/api/ingest"),
    ("https://dash.example.app/api/ingest", "https://dash.example.app/api/ingest"),
    ("https://example.app/team/dash", "https://example.app/team/dash/api/ingest"),
    ("http://localhost:3000", "http://localhost:3000/api/ingest"),
    ("http://127.0.0.1:3000/", "http://127.0.0.1:3000/api/ingest"),
])
def test_ingest_endpoint(base, expected):
    assert snapshot.ingest_endpoint(base) == expected


@pytest.mark.parametrize("base", ["http://dash.example.app", "ftp://dash.example.app", "https://"])
def test_ingest_endpoint_refuses_to_send_a_token_in_the_clear(base):
    with pytest.raises(data.DataError):
        snapshot.ingest_endpoint(base)


class FakeResponse:
    def __init__(self, status=200, text="ok"):
        self.status_code, self.text, self.ok = status, text, 200 <= status < 300


def test_publish_posts_json_with_a_bearer_header(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(snapshot.requests, "post", fake_post)
    snapshot.publish("https://d.example/api/ingest", "s3cret", {"price": 1.0})
    assert seen["url"] == "https://d.example/api/ingest" and seen["body"] == {"price": 1.0}
    assert seen["headers"] == {"Authorization": "Bearer s3cret"} and seen["timeout"]


def test_publish_errors_name_the_status_but_never_the_token(monkeypatch):
    monkeypatch.setattr(snapshot.requests, "post", lambda *a, **k: FakeResponse(401, '{"error":"unauthorized"}'))
    with pytest.raises(data.DataError, match=r"refused the snapshot \(401\)") as excinfo:
        snapshot.publish("https://d.example/api/ingest", "s3cret", {})
    assert "s3cret" not in str(excinfo.value)

    def boom(*a, **k):
        raise requests.ConnectionError("dns failure for https://d.example/?token=s3cret")

    monkeypatch.setattr(snapshot.requests, "post", boom)
    with pytest.raises(data.DataError, match="could not reach") as excinfo:
        snapshot.publish("https://d.example/api/ingest", "s3cret", {})
    assert "s3cret" not in str(excinfo.value)  # only the exception class is reported
