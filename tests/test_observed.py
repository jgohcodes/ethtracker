import json

import pytest

from ethtracker import coinbase, observed
from ethtracker.cli import main

from tests.fakes import FakeDashboard
from tests.test_cli import cloud_env, fake_deribit
from tests.test_coinbase import OtherVenue, ec_key_json

P = "ETH_USDC-PERPETUAL"


def pos(size, entry):
    return [{"product": P, "size": size, "entry": entry}]


def test_a_position_held_at_the_first_run_is_one_trade_first_seen_then_adds_are_priced_from_the_average_entry():
    s = observed.update(observed.load(None), pos(2.0, 2400.0), "2026-10-25T20:00:00+00:00")
    (first,) = observed.open_trades(s, P)
    assert first["first_seen"] and first["after"] is None and first["time"] == "2026-10-25T20:00:00+00:00"
    assert first["size"] == 2.0 and first["price"] == 2400.0
    # one more ETH at 2,100 moves the average entry to (2 * 2400 + 2100) / 3 = 2,300
    s = observed.update(s, pos(3.0, 2300.0), "2026-10-27T00:17:00+00:00")
    _, added = observed.open_trades(s, P)
    assert added["size"] == pytest.approx(1.0) and added["price"] == pytest.approx(2100.0)
    assert not added["first_seen"] and added["time"] == added["after"] == "2026-10-25T20:00:00+00:00"
    assert added["seen_at"] == "2026-10-27T00:17:00+00:00"


def test_a_smaller_position_closes_the_oldest_trades_first_and_a_close_then_reopen_starts_afresh():
    s = observed.update(observed.load(None), pos(2.0, 2400.0), "t1")
    s = observed.update(s, pos(3.0, 2300.0), "t2")
    s = observed.update(s, pos(1.5, 2300.0), "t3")  # sold 1.5: the first trade (2.0) keeps 0.5
    assert [round(t["size"], 6) for t in observed.open_trades(s, P)] == [0.5, 1.0]
    s = observed.update(s, [], "t4")  # closed
    assert observed.open_trades(s, P) == [] and s["positions"][P]["size"] == 0.0
    s = observed.update(s, pos(1.0, 2500.0), "t5")
    (t,) = observed.open_trades(s, P)
    assert t["price"] == 2500.0 and t["after"] == "t4" and not t["first_seen"]


def test_a_flip_through_zero_closes_the_long_and_opens_a_short():
    s = observed.update(observed.load(None), pos(2.0, 2400.0), "t1")
    s = observed.update(s, pos(-1.0, 2350.0), "t2")
    (t,) = observed.open_trades(s, P)
    assert t["size"] == -1.0 and t["price"] == 2350.0


def test_an_unreadable_or_foreign_record_starts_empty_and_the_record_round_trips():
    for text in (None, "", "not json", json.dumps({"v": 99}), "[]"):
        assert observed.load(text) == {"v": observed.SCHEMA, "positions": {}, "trades": []}
    s = observed.update(observed.load(None), pos(2.0, 2400.0), "t1")
    assert observed.load(observed.dump(s)) == s


def test_the_daily_job_dates_trades_from_the_position_it_saw_last_and_keeps_them_out_of_the_logs(capsys, monkeypatch, candles_csv):
    text, _ = ec_key_json()
    monkeypatch.setenv(coinbase.ENV, text)
    client = OtherVenue()
    client.orders = False  # the market whose trades the API does not give
    monkeypatch.setattr(coinbase, "Client", lambda key: client)
    cloud_env(monkeypatch)
    dash = FakeDashboard().install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    (trade,) = dash.snapshots[-1]["account"]["positions"][0]["trades"]
    assert trade["first_seen"] and trade["signals"]["score"] is not None
    assert dash.snapshots[-1]["account"]["positions"][0]["trades_source"] == "observed"
    stored = json.loads(dash.state["trades"])
    assert stored["positions"][P]["size"] == pytest.approx(4.5321)
    out = capsys.readouterr()
    assert "stored the exchange position" in out.out
    for secret in ("4.5321", "2311", "1262"):
        assert secret not in out.out + out.err, secret


def test_a_dashboard_that_refuses_the_record_only_warns(capsys, monkeypatch, candles_csv):
    text, _ = ec_key_json()
    monkeypatch.setenv(coinbase.ENV, text)
    client = OtherVenue()
    client.orders = False
    monkeypatch.setattr(coinbase, "Client", lambda key: client)
    cloud_env(monkeypatch)
    dash = FakeDashboard().install(monkeypatch)
    real_put = dash.put

    def put(url, json=None, headers=None, timeout=None):
        if "name=trades" in url:
            from tests.fakes import FakeResponse
            return FakeResponse(400, {"error": "name must be one of: journal, derivs"})
        return real_put(url, json=json, headers=headers, timeout=timeout)
    monkeypatch.setattr("requests.put", put)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    assert "exchange position not stored" in capsys.readouterr().err
