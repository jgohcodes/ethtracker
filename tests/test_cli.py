import json

import pandas as pd
import pytest

from ethtracker import config, indicators
from ethtracker.cli import main


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

    monkeypatch.setattr("ethtracker.cli.get_provider", lambda name, csv_path=None: Fake())
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


def test_headroom_is_an_error_when_the_rule_is_off(capsys, funded):
    (config.home() / "config.toml").write_text("stress_drop_pct = 0.0\n")
    assert main(["headroom", "--price", "3000"]) == 2
    assert "stress rule is off" in capsys.readouterr().err


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
