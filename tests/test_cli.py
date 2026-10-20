import json

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


def test_size_passes_a_tight_stop_at_15x_and_blocks_a_wide_one(capsys, funded):
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


def test_size_blocks_a_stop_inside_daily_noise_even_when_liquidation_is_safe(capsys, funded, candles, candles_csv):
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


def test_price_from_csv(capsys, candles_csv):
    code, out = run(capsys, "--provider", "csv", "--csv", candles_csv, "price")
    assert code == 0 and float(out) > 0


def test_open_risk_close_history_flow(capsys, funded):
    code, out = run(capsys, "open", "--side", "long", "--entry", 3000, "--stop", 2910, "--leverage", 10,
                    "--target", 3600, "--note", "swing", "--no-history")
    assert code == 0 and "recorded position #1" in out

    code, out = run(capsys, "risk", "--price", 3100)
    assert code == 0 and "#1 LONG" in out and "note: swing" in out and "Open risk to stops" in out

    code, out = run(capsys, "risk", "--price", 2700)
    assert code == 1 and "LIQUIDATED" in out

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
    args = ["open", "--side", "long", "--entry", 3000, "--stop", 2760, "--leverage", 15, "--no-history"]
    code, out = run(capsys, *args)
    assert code == 1 and "Not recorded" in out
    assert json.loads((config.home() / "state.json").read_text())["positions"] == []
    code, out = run(capsys, *args, "--force")
    assert code == 0
    note = json.loads((config.home() / "state.json").read_text())["positions"][0]["note"]
    assert "FORCED" in note


def test_closing_beyond_liquidation_records_the_whole_margin(capsys, funded):
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
