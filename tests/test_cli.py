import json

import pandas as pd
import pytest

from ethtracker import config, data, derivs, indicators, snapshot
from ethtracker.cli import main
from tests.fakes import FakeDashboard

INSTRUMENT = config.Config().derivs_instrument  # the linear USDC perpetual


def run(capsys, *argv) -> tuple[int, str]:
    code = main([str(a) for a in argv])
    return code, capsys.readouterr().out


@pytest.fixture
def funded(capsys):
    main(["init", "--equity", "10000"])
    capsys.readouterr()


@pytest.fixture
def isolated(funded):
    (config.home() / "config.toml").write_text('margin_mode = "isolated"\n')


@pytest.fixture
def isolated_no_stress(funded):
    """Isolated margin without the 40% stress rule, to exercise the stop-vs-liquidation rule on its own."""
    (config.home() / "config.toml").write_text('margin_mode = "isolated"\nstress_drop_pct = 0.0\n')


def test_init_writes_config_and_sets_equity(capsys):
    code, out = run(capsys, "init", "--equity", 10000)
    assert code == 0 and "equity set" in out
    assert (config.home() / "config.toml").exists()
    code, out = run(capsys, "equity")
    assert "$10,000.00" in out


def test_size_without_equity_is_an_error(capsys):
    code = main(["size", "--side", "long", "--entry", "3000", "--stop", "2900", "--leverage", "10", "--no-history"])
    assert code == 2
    assert "equity is not set" in capsys.readouterr().err


def test_size_passes_a_tight_stop_at_15x_and_blocks_a_wide_one_under_isolated_margin(capsys, isolated_no_stress):
    base = ["size", "--side", "long", "--entry", 3000, "--leverage", 15, "--no-history"]
    code, out = run(capsys, *base, "--stop", 2910)
    assert code == 0 and "VERDICT: PASS" in out and "Leverage ladder" in out
    code, out = run(capsys, *base, "--stop", 2760)
    assert code == 1 and "VERDICT: BLOCKED" in out and "BEFORE your stop" in out


def test_size_prints_history_from_csv(capsys, funded, candles, candles_csv):
    last = float(candles["close"].iloc[-1])
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "size", "--side", "long",
                    "--stop", round(last * 0.95, 2), "--leverage", 10, "--target", round(last * 1.2, 2),
                    "--history-days", 500)
    assert "rolling windows" in out and "your liquidation" in out
    assert "Price scenarios" in out


def test_size_blocks_a_stop_inside_daily_noise_even_when_liquidation_is_safe(capsys, isolated, candles, candles_csv):
    last = float(candles["close"].iloc[-1])
    atr_pct = indicators.snapshot(candles).atr_14_pct
    common = ["--provider", "csv", "--csv", candles_csv, "size", "--side", "long", "--history-days", 500]
    tight = round(last * (1 - 0.5 * atr_pct), 2)
    code, out = run(capsys, *common, "--stop", tight, "--leverage", 15)
    assert code == 1 and "14d ATR" in out
    # a 3x-ATR stop clears the noise rule, but only low leverage keeps liquidation far enough beyond it
    wide = round(last * (1 - 3 * atr_pct), 2)
    code, out = run(capsys, *common, "--stop", wide, "--leverage", 2)
    assert code == 0 and "VERDICT: PASS" in out and "Stop width" in out
    code, out = run(capsys, *common, "--stop", wide, "--leverage", 5)
    assert code == 1 and "Max leverage for this stop" in out


def test_snapshot_from_csv(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "snapshot", "--days", 500)
    assert code == 0
    for label in ("SMA 200", "RSI 14", "ATR 14", "Volatility", "rolling windows"):
        assert label in out


def test_sync_fills_the_cache_through_the_wrapper(capsys, monkeypatch):
    from tests.test_store import Fake

    monkeypatch.setattr("ethtracker.cli.get_provider", lambda name, csv_path=None, product="ETH-USD": Fake())
    code, out = run(capsys, "sync", "--days", 50)
    assert code == 0 and "50 daily candles" in out
    assert (config.home() / "candles" / "fake-ETH-USD-1d.csv").exists()


def test_sync_refuses_when_there_is_nothing_to_cache(capsys, candles_csv):
    assert main(["--provider", "csv", "--csv", str(candles_csv), "sync"]) == 2
    assert "sync fills the local cache" in capsys.readouterr().err
    assert main(["--no-cache", "sync"]) == 2


def test_export_writes_candles_with_every_requested_metric(capsys, candles, candles_csv, tmp_path):
    out_path = tmp_path / "out" / "eth_metrics.csv"
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "export", "--out", out_path, "--days", 500)
    assert code == 0 and "wrote 500 rows" in out
    df = pd.read_csv(out_path, index_col="date")
    for column in ("open", "high", "low", "close", "volume", "sma_20", "sma_50", "sma_100", "sma_200", "rsi_14"):
        assert column in df.columns
    assert len(df) == 500
    assert df["sma_100"].iloc[-1] == pytest.approx(candles["close"].tail(100).mean())
    assert df["sma_200"].iloc[:199].isna().all()


def test_size_shows_the_stress_line_and_that_it_capped_the_size(capsys, funded):
    code, out = run(capsys, "size", "--side", "long", "--entry", 3000, "--stop", 2995, "--leverage", 5, "--no-history")
    assert "Stress       a 40% fall loses 99%" in out and "[size capped by this rule]" in out
    assert "not surviving financially" in out


def test_headroom_reports_room_the_99_percent_loss_and_the_isolated_leverage(capsys, funded):
    code, out = run(capsys, "headroom", "--price", 3000)
    assert code == 0
    assert "up to $24,814 more long notional" in out
    assert "leaves $74: 99% of equity lost" in out


def test_a_book_sized_to_the_limit_breaches_after_a_further_fall(capsys, funded):
    code, out = run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2995, "--leverage", 5,
                    "--qty", 8.27, "--no-history")
    assert code == 0
    code, out = run(capsys, "headroom", "--price", 2400)
    assert "BREACHED" in out and "up to $0 more long notional" in out
    code, out = run(capsys, "risk", "--price", 2400)
    assert code == 1 and "STRESS_BREACH" in out and "margin limit BREACHED" in out
    code, out = run(capsys, "risk", "--price", 3000)
    assert "Whole-book liquidation" in out and "margin limit holds" in out


def test_cross_margin_size_explains_that_leverage_does_not_move_liquidation(capsys, funded):
    code, out = run(capsys, "size", "--side", "long", "--entry", 3000, "--stop", 2760, "--leverage", 15, "--no-history")
    assert code == 0  # the 15x / 8% stop that isolated margin blocks
    assert "Leverage ladder" not in out and "Cross margin: the exchange leverage setting" in out
    assert "cross margin: set by the whole book" in out


def test_isolated_size_shows_the_ladder_and_blocks_leverage_that_a_stress_move_liquidates(capsys, isolated):
    code, out = run(capsys, "size", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10, "--no-history")
    assert code == 1 and "Leverage ladder" in out
    assert "Isolated margin: at 10x" in out and "at most 2.5x" in out


def test_headroom_explains_why_it_does_not_apply_to_isolated_margin(capsys, isolated):
    assert main(["headroom", "--price", "3000"]) == 2
    err = capsys.readouterr().err
    assert "only applies to cross margin" in err and "<= 2.5x" in err


def test_a_bad_margin_mode_is_a_clean_error(capsys, funded):
    (config.home() / "config.toml").write_text('margin_mode = "portfolio"\n')
    assert main(["risk", "--price", "3000"]) == 2
    assert "margin_mode must be" in capsys.readouterr().err


def test_watch_alerts_on_whole_book_flags_once(capsys, funded, candles_csv):
    # csv price is ~$292 against a $3,000 entry, so this book has lost its whole balance
    run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2995, "--leverage", 5, "--qty", 8.27, "--no-history")
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "watch", "--once")
    assert "ALERT portfolio STRESS_BREACH" in out and "ALERT portfolio LIQUIDATED" in out


def test_a_book_smaller_than_equity_says_it_cannot_be_liquidated_not_zero_dollars(capsys, funded):
    run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10, "--no-history")  # ~0.3x equity
    code, out = run(capsys, "risk", "--price", 3000)
    line = next(l for l in out.splitlines() if l.startswith("Whole-book liquidation"))
    assert line.startswith("Whole-book liquidation: none") and "$" not in line
    code, out = run(capsys, "headroom", "--price", 3000)
    assert any(l.startswith("Whole-book liquidation: none") for l in out.splitlines())


def test_headroom_is_an_error_when_the_rule_is_off(capsys, funded):
    (config.home() / "config.toml").write_text("stress_drop_pct = 0.0\n")
    assert main(["headroom", "--price", "3000"]) == 2
    assert "stress rule is off" in capsys.readouterr().err


@pytest.fixture
def derivs_csv(tmp_path):
    path = tmp_path / "derivs.csv"
    idx = pd.date_range("2026-09-01", periods=40, freq="D", tz="UTC")
    frame = pd.DataFrame({"open_interest": [100.0 + i for i in range(40)], "funding_8h": [0.0001] * 39 + [0.0007]}, index=idx)
    frame.index = frame.index.strftime("%Y-%m-%d")
    frame.to_csv(path, index_label="date")
    return path


def open_long(capsys, entry):
    code, _ = run(capsys, "open", "--side", "long", "--entry", entry, "--stop", entry * 0.96, "--leverage", 5, "--no-history")
    assert code == 0


def test_signals_prints_levels_both_groups_and_the_data_gaps(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "signals", "--days", 500)
    assert code == 0
    for text in ("Levels", "AVWAP from the low", "point of control", "Technical", "Net by family", "Risk flags:",
                 "note: no derivatives data"):
        assert text in out
    assert "Derivatives  (" not in out


def test_signals_shows_derivatives_flagged_unvalidated_when_given_a_csv(capsys, candles_csv, derivs_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "--derivs-csv", derivs_csv, "signals")
    assert "Derivatives" in out
    assert "Funding 0.070% per 8h" in out and "crowded" in out and "(unvalidated)" in out
    assert "percentile of its 40-day history" in out


def test_signals_accepts_an_anchor_date(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "signals", "--anchor", "2025-06-01")
    assert "the chosen anchor (2025-06-01)" in out
    assert main(["--provider", "csv", "--csv", str(candles_csv), "signals", "--anchor", "not-a-date"]) == 2


def test_advise_with_no_positions(capsys, funded, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "advise")
    assert code == 0 and "No open positions" in out


def test_advise_exits_a_position_through_its_stop_with_both_cases(capsys, funded, candles_csv):
    open_long(capsys, 3000)  # the csv price is ~$292
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "advise")
    assert code == 1 and "ACTION: EXIT" in out
    assert "why: Stop" in out and "Bull case" in out and "Bear case" in out


def test_advise_trails_a_deep_in_the_money_position_when_signals_are_bearish(capsys, funded, candles_csv):
    open_long(capsys, 250)  # csv price ~$292 is ~17% above entry; the synthetic tape is in a downtrend
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "advise")
    assert ("ACTION: TIGHTEN" in out and code == 0) or ("ACTION: EXIT" in out and code == 1)
    assert "TAKE_PARTIAL" not in out  # flags alone never sell part of a position
    assert "Deep in the money" in out


def test_advise_tightens_a_modest_profit_when_net_signals_are_bearish(capsys, funded, candles_csv):
    open_long(capsys, 285)  # ~+2% in profit, not deep
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "advise")
    assert "ACTION: TIGHTEN" in out and code == 0


def test_record_appends_a_snapshot_and_skips_a_repeat(capsys, funded, monkeypatch):
    row = derivs.parse_deribit_ticker({"result": {"open_interest": 1000.0, "funding_8h": 0.0002, "timestamp": 1760961600000}})
    monkeypatch.setattr(derivs, "fetch_deribit_snapshot", lambda instrument: row)
    code, out = run(capsys, "record", "--once")
    assert code == 0 and INSTRUMENT in out and "OI 1,000" in out and "0.0200%" in out
    path = derivs.recorded_path(config.home() / "derivs", INSTRUMENT)
    assert len(path.read_text().splitlines()) == 2
    code, out = run(capsys, "record", "--once")
    assert "duplicate timestamp" in out and len(path.read_text().splitlines()) == 2


def test_recorded_history_feeds_the_signals_automatically(capsys, funded, candles_csv):
    path = derivs.recorded_path(config.home() / "derivs", INSTRUMENT)
    derivs.append_snapshot(path, derivs.parse_deribit_ticker({"result": {"open_interest": 1000.0, "funding_8h": 0.0008, "timestamp": 1760961600000}}))
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "signals")
    assert "Funding 0.080% per 8h" in out and "longs paying heavily" in out


def test_signals_study_prints_buckets_and_the_caveats(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "signals-study", "--days", 500)
    assert code == 0
    for text in ("all days", "2+ flags", "episodes", "horizon 14d", "Windows overlap", "not tested here"):
        assert text in out


class SequenceProvider:
    """Replays a list of spot prices over fixed candles, to drive `watch` through several polls."""

    name = "fake"

    def __init__(self, prices, candles):
        self.prices, self.frame, self.polls = list(prices), candles, 0

    def spot(self):
        price = self.prices[min(self.polls, len(self.prices) - 1)]
        self.polls += 1
        return price

    def candles(self, days):
        return self.frame.tail(days)


def watch_polls(monkeypatch, capsys, candles, prices, *extra):
    provider = SequenceProvider(prices, candles)
    monkeypatch.setattr("ethtracker.cli.get_provider", lambda name, csv_path=None, product="ETH-USD": provider)
    sleeps = []

    def sleeper(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= len(prices):
            raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", sleeper)
    code = main(["watch", "--interval", "1", *extra])
    return code, capsys.readouterr().out


def test_watch_announces_an_action_change_once_not_on_every_poll(capsys, monkeypatch, funded, candles):
    open_long(capsys, 250)
    code, out = watch_polls(monkeypatch, capsys, candles, [245.0, 245.0, 291.0])
    assert code == 0
    assert out.count("ACTION CHANGE") == 1  # HOLD while underwater, then deep in the money
    change = next(line for line in out.splitlines() if "ACTION CHANGE" in line)
    assert "HOLD ->" in change and ("TIGHTEN" in change or "EXIT" in change)
    assert "ACTION:" in out and "Bull case" in out and "Bear case" in out
    polls = [line for line in out.splitlines() if line.startswith("  #1 long")]
    assert len(polls) == 3 and polls[0].endswith("-> HOLD") and "->" in polls[2]


def test_watch_stays_quiet_about_a_position_that_holds(capsys, monkeypatch, funded, candles):
    open_long(capsys, 250)
    code, out = watch_polls(monkeypatch, capsys, candles, [245.0, 246.0, 247.0])
    assert "ACTION CHANGE" not in out and out.count("-> HOLD") == 3


def test_watch_alerts_immediately_when_the_first_sighting_is_an_exit(capsys, monkeypatch, funded, candles):
    open_long(capsys, 3000)
    code, out = watch_polls(monkeypatch, capsys, candles, [291.0])
    assert "ACTION CHANGE #1: new -> EXIT" in out


def test_watch_record_appends_a_derivatives_snapshot_each_poll(capsys, monkeypatch, funded, candles):
    rows = iter(derivs.parse_deribit_ticker({"result": {"open_interest": 1000.0 + i, "funding_8h": 0.0001, "timestamp": 1760961600000 + i * 3600000}}) for i in range(5))
    monkeypatch.setattr(derivs, "fetch_deribit_snapshot", lambda instrument: next(rows))
    watch_polls(monkeypatch, capsys, candles, [291.0, 292.0], "--record")
    path = derivs.recorded_path(config.home() / "derivs", INSTRUMENT)
    assert len(path.read_text().splitlines()) == 3  # header plus one row per poll


def test_watch_survives_a_failed_derivatives_record(capsys, monkeypatch, funded, candles):
    def boom(instrument):
        raise data.DataError("offline")

    monkeypatch.setattr(derivs, "fetch_deribit_snapshot", boom)
    code, out = watch_polls(monkeypatch, capsys, candles, [291.0], "--record")
    assert code == 0 and "ETHUSD" in out


def test_dashboard_json_writes_strict_json_for_the_book(capsys, funded, candles_csv, tmp_path):
    open_long(capsys, 285)
    path = tmp_path / "out" / "dash.json"
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "dashboard-json", "--out", path)
    assert code == 0 and "1 position(s)" in out
    snap = json.loads(path.read_text())
    assert snap["positions"][0]["action"] in ("HOLD", "TIGHTEN", "TAKE_PARTIAL", "EXIT")
    assert snap["signals"] and snap["equity"]["wallet"] == 10_000


def test_dashboard_json_prints_to_stdout_without_an_out_path(capsys, funded, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "dashboard-json")
    assert code == 0 and json.loads(out)["positions"] == []


def test_publish_once_posts_the_snapshot_pushes_the_journal_and_never_echoes_the_token(capsys, monkeypatch, funded, candles_csv):
    open_long(capsys, 285)
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "s3cret-ingest-token")
    dash = FakeDashboard(token="s3cret-ingest-token").install(monkeypatch)
    code = main(["--provider", "csv", "--csv", str(candles_csv), "publish", "--url", "http://localhost:3000", "--once"])
    captured = capsys.readouterr()
    assert code == 0 and "published ETHUSD" in captured.out and "#1 " in captured.out
    assert "s3cret-ingest-token" not in captured.out + captured.err
    (snap,) = dash.snapshots
    assert snap["positions"][0]["id"] == 1 and snap["basis"] == "close"
    assert snap["publisher"] == "local" and snap["refresh_seconds"] == 3600  # after --once the hourly job is the cadence
    assert ("POST", "http://localhost:3000/api/ingest") in dash.calls
    journal = json.loads(dash.state["journal"])
    assert journal == json.loads((config.home() / "state.json").read_text())


def test_publish_uses_the_derivatives_history_kept_by_the_dashboard(capsys, monkeypatch, funded, candles_csv):
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "t")
    history = "time,open_interest,funding_8h\n2026-10-20T00:20:00+00:00,100,0.0007\n"
    dash = FakeDashboard(state={"derivs": history}).install(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "publish", "--url", "https://d.example", "--once"]) == 0
    funding = next(s for s in dash.snapshots[0]["signals"] if s["name"] == "funding")
    assert "0.070%" in funding["headline"]


def test_publish_reads_the_url_from_the_environment(capsys, monkeypatch, funded, candles_csv):
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "t")
    monkeypatch.setenv("ETHTRACKER_DASHBOARD_URL", "https://dash.example.app")
    dash = FakeDashboard().install(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "publish", "--once"]) == 0
    assert ("POST", "https://dash.example.app/api/ingest") in dash.calls


def test_publish_needs_a_url_and_a_token_and_refuses_plain_http(capsys, monkeypatch, funded, candles_csv):
    base = ["--provider", "csv", "--csv", str(candles_csv), "publish", "--once"]
    monkeypatch.delenv("ETHTRACKER_INGEST_TOKEN", raising=False)
    monkeypatch.delenv("ETHTRACKER_DASHBOARD_URL", raising=False)
    assert main(base) == 2 and "ETHTRACKER_DASHBOARD_URL" in capsys.readouterr().err
    assert main([*base, "--url", "https://d.example"]) == 2 and "ETHTRACKER_INGEST_TOKEN" in capsys.readouterr().err
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "t")
    assert main([*base, "--url", "http://d.example"]) == 2 and "refusing to send a token" in capsys.readouterr().err


def test_publish_first_failure_is_fatal_but_later_ones_are_warnings(capsys, monkeypatch, funded, candles_csv):
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "t")
    base = ["--provider", "csv", "--csv", str(candles_csv), "publish", "--url", "https://d.example", "--interval", "1"]
    FakeDashboard(unreachable=True).install(monkeypatch)
    assert main([*base, "--once"]) == 2
    assert "could not reach" in capsys.readouterr().err

    dash = FakeDashboard(fail_posts_after=1).install(monkeypatch)  # the first publish works, the second cannot
    sleeps = []

    def sleeper(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", sleeper)
    assert main(base) == 0
    captured = capsys.readouterr()
    assert captured.out.count("published ETHUSD") == 1 and "not published this round" in captured.err
    assert len(dash.snapshots) == 1


def cloud_journal(tmp_path, equity=123_456.0, entry=285.0) -> str:
    from ethtracker.journal import Journal

    j = Journal(tmp_path / "cloud" / "state.json")
    j.set_equity(equity)
    j.open(side="long", entry=entry, qty=7.0, leverage=5.0, stop=entry * 0.96, target=None, mmr=0.005,
           initial_risk_usd=100.0, note="private note")
    j.save()
    return (tmp_path / "cloud" / "state.json").read_text()


def fake_deribit(monkeypatch, *, fail=False):
    if fail:
        def boom(instrument):
            raise data.DataError("GET https://www.deribit.com/api/v2/public/ticker failed: 451")
        monkeypatch.setattr(derivs, "fetch_deribit_snapshot", boom)
        return
    row = derivs.parse_deribit_ticker({"result": {"open_interest": 1200.0, "funding_8h": 0.0006}})
    row["time"] = "2026-10-24T00:20:00+00:00"  # the day after the stored history's last row
    monkeypatch.setattr(derivs, "fetch_deribit_snapshot", lambda instrument: row)


def cloud_env(monkeypatch, url="https://dash.example"):
    monkeypatch.setenv("ETHTRACKER_DASHBOARD_URL", url)
    monkeypatch.setenv("ETHTRACKER_INGEST_TOKEN", "t")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


def test_cloud_run_publishes_from_the_stored_journal_and_grows_the_derivatives_history(capsys, monkeypatch, candles_csv, tmp_path):
    cloud_env(monkeypatch)
    history = "time,open_interest,funding_8h,current_funding,mark_price,index_price\n2026-10-23T00:20:00+00:00,1000,0.0001,,,\n"
    dash = FakeDashboard(state={"journal": cloud_journal(tmp_path), "derivs": history}).install(monkeypatch)
    fake_deribit(monkeypatch)
    code = main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    (snap,) = dash.snapshots
    assert len(snap["positions"]) == 1 and snap["equity"]["wallet"] == 123_456.0
    assert snap["basis"] == "close" and snap["refresh_seconds"] == 3600 and snap["publisher"] == "cloud-run"
    assert any(s["name"] == "funding" and "0.060%" in s["headline"] for s in snap["signals"])
    rows = dash.state["derivs"].strip().splitlines()
    assert len(rows) == 3 and rows[0].startswith("time,")  # header, yesterday's row, today's row
    assert "journal: pulled from the dashboard" in captured.out
    assert "published the snapshot and the derivatives history" in captured.out
    assert not (config.home() / "state.json").exists()  # the local journal is never touched


def test_cloud_run_output_is_safe_for_public_logs(capsys, monkeypatch, candles_csv, tmp_path):
    cloud_env(monkeypatch)
    FakeDashboard(state={"journal": cloud_journal(tmp_path)}).install(monkeypatch)
    fake_deribit(monkeypatch, fail=True)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    for secret in ("123,456", "123456", "285", "private note", "7.0000", "dash.example/api/state", " t\n", "position"):
        assert secret not in logs, secret


def test_cloud_run_marks_its_snapshots_as_from_github_actions_there(capsys, monkeypatch, candles_csv):
    cloud_env(monkeypatch)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    dash = FakeDashboard().install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    assert dash.snapshots[0]["publisher"] == "github-actions"


def test_cloud_run_with_no_journal_yet_shows_the_market_only(capsys, monkeypatch, candles_csv):
    cloud_env(monkeypatch)
    dash = FakeDashboard().install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    snap = dash.snapshots[0]
    assert snap["positions"] == [] and snap["equity"] is None and snap["signals"]
    assert "journal: none stored yet" in capsys.readouterr().out
    assert dash.state["derivs"].count("\n") == 2  # header plus the first recorded day


def test_cloud_run_survives_a_failed_derivatives_fetch_and_keeps_the_old_history(capsys, monkeypatch, candles_csv):
    cloud_env(monkeypatch)
    dash = FakeDashboard(state={"derivs": "time,open_interest,funding_8h\n2026-10-23T00:20:00+00:00,1000,0.0001\n"}).install(monkeypatch)
    fake_deribit(monkeypatch, fail=True)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    captured = capsys.readouterr()
    assert "derivatives snapshot unavailable" in captured.err and "published the snapshot" in captured.out
    assert ("PUT", "https://dash.example/api/state?name=derivs") not in dash.calls  # nothing new to store
    assert dash.snapshots


def test_cloud_run_rejects_a_corrupt_stored_journal(capsys, monkeypatch, candles_csv):
    cloud_env(monkeypatch)
    FakeDashboard(state={"journal": "{not json"}).install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 2
    assert "journal stored on the dashboard is not valid" in capsys.readouterr().err


def test_cloud_run_needs_credentials_unless_it_is_a_dry_run(capsys, monkeypatch, candles_csv, tmp_path):
    monkeypatch.delenv("ETHTRACKER_DASHBOARD_URL", raising=False)
    monkeypatch.delenv("ETHTRACKER_INGEST_TOKEN", raising=False)
    dash = FakeDashboard().install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 2
    out = tmp_path / "snap.json"
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run", "--dry-run", "--out", str(out)]) == 0
    assert "dry run" in capsys.readouterr().out
    assert dash.calls == [] and json.loads(out.read_text())["basis"] == "close"


def test_cloud_run_fails_loudly_when_publishing_fails(capsys, monkeypatch, candles_csv):
    cloud_env(monkeypatch)
    FakeDashboard(fail_posts_after=0).install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 2
    assert "could not reach" in capsys.readouterr().err


def test_study_from_csv_prints_trades_summary_and_caveats(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "study", "--trigger", 10,
                    "--stop-atr", 2, "--target-r", 2, "--leverage", 3, "--days", 500)
    assert code == 0
    for text in ("Rule: long", "winners", "R multiples", "Buy-and-hold", "Not tuned"):
        assert text in out


def test_study_needs_exactly_one_stop_definition(capsys, candles_csv):
    with pytest.raises(SystemExit):
        main(["--provider", "csv", "--csv", str(candles_csv), "study", "--leverage", "3"])
    with pytest.raises(SystemExit):
        main(["--provider", "csv", "--csv", str(candles_csv), "study", "--leverage", "3", "--stop-pct", "5", "--stop-atr", "2"])


def test_snapshot_shows_pullback_gauges(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "snapshot", "--days", 500)
    assert "Pullback" in out and "below the 30d high" in out


def test_price_from_csv(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "price")
    assert code == 0 and float(out) > 0


def test_open_risk_close_history_flow(capsys, funded):
    code, out = run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10,
                    "--target", 3600, "--note", "swing", "--no-history")
    assert code == 0 and "recorded position #1" in out

    code, out = run(capsys, "risk", "--price", 3100)
    assert code == 0 and "#1 LONG" in out and "note: swing" in out and "Open risk to stops" in out

    code, out = run(capsys, "risk", "--price", 2700)  # past the stop; a 0.3x book is nowhere near liquidation
    assert code == 1 and "STOP_HIT" in out and "LIQUIDATED" not in out

    code, out = run(capsys, "close", 1, "--price", 3300)
    assert code == 0 and "realized" in out
    state = json.loads((config.home() / "state.json").read_text())
    assert 10_300 < state["equity"] < 10_330
    assert state["peak_equity"] == state["equity"]

    code, out = run(capsys, "history")
    assert "1 trades, 1 winners" in out
    code, out = run(capsys, "risk", "--price", 3300)
    assert "No open positions" in out


def test_open_rejects_a_rule_breaking_plan_unless_forced(capsys, funded):
    args = ["open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 20, "--no-history"]
    code, out = run(capsys, *args)
    assert code == 1 and "Not recorded" in out
    assert json.loads((config.home() / "state.json").read_text())["positions"] == []
    code, out = run(capsys, *args, "--force")
    assert code == 0
    note = json.loads((config.home() / "state.json").read_text())["positions"][0]["note"]
    assert "FORCED" in note


def test_closing_beyond_liquidation_records_the_whole_margin_under_isolated_margin(capsys, isolated_no_stress):
    run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10, "--no-history")
    state = json.loads((config.home() / "state.json").read_text())
    margin = state["positions"][0]["qty"] * 3000 / 10
    code, out = run(capsys, "close", 1, "--price", 2500)
    assert "full margin" in out
    assert json.loads((config.home() / "state.json").read_text())["equity"] == pytest.approx(10_000 - margin)


def test_watch_once_reports_price_and_position_flags(capsys, funded, candles_csv):
    run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10, "--no-history")
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "watch", "--once", "--above", 1)
    assert code == 0 and "ETHUSD" in out and "#1 long" in out


def test_close_unknown_id_is_a_clean_error(capsys, funded):
    assert main(["close", "7", "--price", "3000"]) == 2
    assert "no position with id 7" in capsys.readouterr().err
