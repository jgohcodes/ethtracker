import pytest

from ethtracker import alerts, config
from ethtracker.journal import Journal


def open_one(journal, **overrides):
    fields = dict(side="long", entry=3000.0, qty=1.0, leverage=10.0, stop=2900.0, target=None,
                  mmr=0.005, initial_risk_usd=100.0)
    fields.update(overrides)
    return journal.open(**fields)


def test_journal_round_trips_through_disk(tmp_path):
    path = tmp_path / "state.json"
    j = Journal(path)
    j.set_equity(10_000)
    open_one(j, note="first")
    j.save()
    again = Journal(path)
    assert again.equity == 10_000 and again.peak_equity == 10_000
    assert [p.note for p in again.open_positions()] == ["first"]
    assert again.next_id == 2


def test_closing_updates_equity_and_peak_only_upwards(tmp_path):
    j = Journal(tmp_path / "state.json")
    j.set_equity(10_000)
    win, loss = open_one(j), open_one(j)
    j.close(win.id, 3300.0, 300.0)
    assert (j.equity, j.peak_equity) == (10_300, 10_300)
    j.close(loss.id, 2900.0, -100.0)
    assert (j.equity, j.peak_equity) == (10_200, 10_300)
    assert len(j.closed_positions()) == 2 and not j.open_positions()


def test_closing_twice_or_unknown_id_fails(tmp_path):
    j = Journal(tmp_path / "state.json")
    pos = open_one(j)
    j.close(pos.id, 3100.0, 50.0)
    with pytest.raises(ValueError):
        j.close(pos.id, 3100.0, 50.0)
    with pytest.raises(KeyError):
        j.get(99)


def test_set_equity_resets_the_drawdown_baseline(tmp_path):
    j = Journal(tmp_path / "state.json")
    j.set_equity(10_000)
    j.set_equity(7_000)  # withdrawal
    assert j.peak_equity == 7_000


def test_crossed_fires_on_a_crossing_in_either_direction_but_not_the_first_poll():
    assert not alerts.crossed(None, 3000, 3000)
    assert alerts.crossed(2990, 3010, 3000)
    assert alerts.crossed(3010, 2990, 3000)
    assert alerts.crossed(2990, 3000, 3000)
    assert not alerts.crossed(3001, 3010, 3000)


def test_new_flags_reports_each_flag_once_and_refires_after_clearing():
    seen: set = set()
    assert alerts.new_flags(seen, {(1, "STOP_HIT")}) == {(1, "STOP_HIT")}
    assert alerts.new_flags(seen, {(1, "STOP_HIT")}) == set()
    assert alerts.new_flags(seen, set()) == set()
    assert alerts.new_flags(seen, {(1, "STOP_HIT")}) == {(1, "STOP_HIT")}


def test_default_config_round_trips_and_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(config.render_default_toml())
    assert config.load(path) == config.Config()
    path.write_text("risk_per_trade_pct = 2.0\nmax_levrage = 10\n")
    with pytest.raises(ValueError, match="max_levrage"):
        config.load(path)


def test_missing_config_falls_back_to_defaults(tmp_path):
    assert config.load(tmp_path / "absent.toml") == config.Config()
