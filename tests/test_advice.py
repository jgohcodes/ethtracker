from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ethtracker import advice, indicators, risk
from ethtracker.config import Config
from ethtracker.signals import Signal, SignalSet

CFG = Config()
NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


def position(side="long", entry=3000.0, stop=2900.0, days=3.0) -> risk.Position:
    return risk.Position(id=1, side=side, entry=entry, qty=1.0, leverage=5.0, stop=stop, target=None,
                         opened_at=(NOW - timedelta(days=days)).isoformat(), mmr=0.005, initial_risk_usd=100.0)


def flat_candles(price=3000.0, n=80) -> pd.DataFrame:
    idx = pd.date_range(end=pd.Timestamp(NOW).normalize(), periods=n, freq="D")
    close = pd.Series(price, index=idx)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})


def sig(name, bias=0, risk_flag=False, headline=None, unvalidated=False) -> Signal:
    return Signal(name, bias, risk_flag, headline or name, unvalidated)


def run(pos, price, signal_list, portfolio_flags=(), cfg=CFG, df=None, plan_floor=None):
    sigs = SignalSet(price=price, signals=tuple(signal_list), levels={}, notes=())
    status = risk.mark_position(cfg, pos, price, NOW)
    df = indicators.with_price(flat_candles() if df is None else df, price)
    return advice.assess(cfg, pos, status, sigs, df, portfolio_flags, plan_floor=plan_floor)


BULL = [sig("trend", 1, headline="uptrend"), sig("poc", 1, headline="above POC")]
RISKY = lambda n: [sig(f"risk{i}", 0, True, headline=f"risk flag {i}") for i in range(n)]


def test_a_breached_stop_is_an_exit_before_anything_else():
    a = run(position(), 2850, BULL)
    assert a.action == "EXIT" and a.suggested_fraction == 1.0 and a.suggested_stop is None
    assert any("breached" in r for r in a.reasons)


def test_the_time_stop_is_an_exit():
    a = run(position(days=15), 3010, BULL)
    assert a.action == "EXIT" and any("time stop" in r for r in a.reasons)


def test_a_book_near_liquidation_exits_and_a_stress_breach_cuts_size():
    assert run(position(), 3050, BULL, portfolio_flags=("LIQ_DANGER",)).action == "EXIT"
    cut = run(position(), 3050, BULL, portfolio_flags=("STRESS_BREACH",))
    assert cut.action == "TAKE_PARTIAL" and cut.suggested_fraction == 0.5


def test_deep_in_the_money_with_enough_risk_flags_is_an_exit():
    a = run(position(), 3360, BULL + RISKY(2))  # +12% from entry
    assert a.deep_itm and a.action == "EXIT" and a.risk_flags == 2


def test_deep_in_the_money_with_one_risk_flag_trails_but_does_not_sell():
    a = run(position(), 3360, BULL + RISKY(1))
    assert a.action == "TIGHTEN" and a.suggested_fraction == 0.0
    assert a.suggested_stop is not None and 2900 < a.suggested_stop < 3360


def test_deep_in_the_money_with_a_bearish_net_score_trails():
    a = run(position(), 3360, [sig("trend", -1), sig("poc", -1), sig("rsi", 1)])
    assert a.net_score == -1 and a.risk_flags == 0 and a.action == "TIGHTEN" and a.suggested_fraction == 0.0


def test_deep_in_the_money_with_everything_supportive_only_trails():
    a = run(position(), 3360, BULL)
    assert a.deep_itm and a.action == "TIGHTEN" and a.suggested_fraction == 0.0
    assert any("bull case intact" in r for r in a.reasons)
    assert a.suggested_stop is not None and a.suggested_stop > 2900


def test_risk_flags_alone_never_sell_part_of_a_position():
    a = run(position(), 3090, [sig("trend", -1)] + RISKY(2))  # +3%, not deep, flags stacking
    assert not a.deep_itm and a.action == "TIGHTEN" and a.suggested_fraction == 0.0


def test_in_profit_with_a_single_risk_flag_tightens():
    a = run(position(), 3090, BULL + RISKY(1))
    assert a.action == "TIGHTEN"


def test_in_profit_with_supportive_signals_holds():
    a = run(position(), 3090, BULL)
    assert a.action == "HOLD" and a.suggested_stop is None


def test_an_underwater_position_with_deteriorating_signals_is_told_to_honour_its_stop():
    a = run(position(), 2960, [sig("trend", -1), sig("poc", -1)] + RISKY(2))
    assert a.action == "HOLD" and any("honour the stop" in r for r in a.reasons)


def test_the_exit_threshold_is_configurable():
    strict = replace(CFG, exit_risk_flags=1)
    assert run(position(), 3360, BULL + RISKY(1), cfg=strict).action == "EXIT"
    loose = replace(CFG, deep_itm_pct=20)
    assert not run(position(), 3360, BULL + RISKY(1), cfg=loose).deep_itm


def test_a_short_mirrors_the_long():
    short = position(side="short", entry=3000.0, stop=3100.0)
    bearish_for_price = [sig("trend", -1, headline="downtrend"), sig("poc", -1, headline="below POC")]
    deep = run(short, 2640, bearish_for_price + RISKY(2))  # price down 12%: deep in the money for a short
    assert deep.deep_itm and deep.action == "EXIT" and deep.move == pytest.approx(0.12)
    assert deep.bull_case[1:] == ("downtrend", "below POC")  # bearish-for-price signals support a short
    against = run(short, 2640, [sig("trend", 1), sig("poc", 1), sig("rsi", -1)])
    assert against.action == "TIGHTEN" and against.net_score == -1
    stopped = run(short, 3150, bearish_for_price)
    assert stopped.action == "EXIT"


def test_bull_and_bear_cases_list_aligned_opposing_and_risk_signals_with_position_context():
    signal_list = [
        sig("trend", 1, headline="uptrend"),
        sig("poc", -1, headline="below POC"),
        sig("funding", -1, True, headline="funding hot", unvalidated=True),
        sig("volatility", 0, True, headline="volatility spiking"),
        sig("rsi", 0, headline="RSI neutral"),
    ]
    a = run(position(), 3360, signal_list)
    assert a.bull_case == ("In profit: +12.0% from entry", "uptrend")
    assert a.bear_case[0].startswith("Deep in the money")
    assert "below POC" in a.bear_case
    assert "funding hot (unvalidated)" in a.bear_case  # counted once even though it is both bearish and a risk flag
    assert a.bear_case.count("funding hot (unvalidated)") == 1
    assert "volatility spiking" in a.bear_case
    assert "RSI neutral" not in a.bull_case + a.bear_case


def test_correlated_signals_in_one_family_count_once_in_the_net_score():
    stacked = [sig("trend", -1), sig("long_trend", -1), sig("avwap_low", -1), sig("poc", -1)]
    one_family = [replace(g, family="levels") for g in stacked]
    separate = run(position(), 3090, stacked)
    pooled = run(position(), 3090, one_family + [sig("rsi", 1, headline="RSI oversold")])
    assert separate.net_score == -4
    assert pooled.net_score == 0  # four bearish level signals vote once (-1), the RSI family votes +1


def test_a_downtrend_does_not_force_a_deep_winner_out_on_stacked_level_signals_alone():
    levels = [replace(sig(n, -1), family="levels") for n in ("trend", "long_trend", "avwap_low", "poc")]
    momentum = [sig("rsi", 1, headline="RSI oversold"), sig("extension", 1, headline="washed out")]
    a = run(position(), 3360, levels + [replace(g, family="momentum") for g in momentum])
    assert a.net_score == 0 and a.action == "TIGHTEN"  # votes tie 1-1, so it is not a TAKE_PARTIAL


def test_an_underwater_position_says_so_in_the_bear_case():
    a = run(position(), 2950, BULL)
    assert a.bear_case[0].startswith("Underwater") and a.move < 0


def trending_candles(price_end=3300.0) -> pd.DataFrame:
    """Flat at 3000 with a one-day rally to ``price_end``: a known ATR and a known best price."""
    df = flat_candles(3000.0)
    df.iloc[-1] = [3000.0, price_end, 3000.0, price_end, 1.0]
    return df


def test_the_trailing_stop_is_a_chandelier_behind_the_best_price_and_never_loosens():
    df = trending_candles(3300.0)
    pos = position(days=1)
    atr = float(indicators.atr(df, 14).iloc[-1])
    expected = 3300.0 - CFG.trail_atr_multiple * atr
    stop = advice.trailing_stop(CFG, pos, df, 3300.0)
    assert stop == pytest.approx(expected) and stop > pos.stop
    tight = position(days=1, stop=3290.0)
    assert advice.trailing_stop(CFG, tight, df, 3300.0) is None  # the existing stop is already tighter


def test_breakeven_is_offered_once_a_full_atr_in_profit_and_the_tighter_candidate_wins():
    df = flat_candles(3000.0)
    df.iloc[-1] = [3000.0, 3012.0, 3000.0, 3010.0, 1.0]
    pos = position(days=1, stop=2900.0)
    price = 3010.0
    atr = float(indicators.atr(df, 14).iloc[-1])
    assert price - pos.entry >= atr  # a full ATR in profit
    stop = advice.trailing_stop(CFG, pos, df, price)
    assert stop == pytest.approx(max(pos.entry, 3012.0 - 3 * atr))
    assert stop >= pos.entry  # at least breakeven


def test_a_short_trails_above_the_lowest_price_since_entry():
    df = flat_candles(3000.0)
    df.iloc[-1] = [3000.0, 3000.0, 2700.0, 2700.0, 1.0]
    short = position(side="short", entry=3000.0, stop=3100.0, days=1)
    atr = float(indicators.atr(df, 14).iloc[-1])
    stop = advice.trailing_stop(CFG, short, df, 2700.0)
    assert stop == pytest.approx(2700.0 + CFG.trail_atr_multiple * atr) and stop < 3100.0


def test_no_trail_without_enough_history_for_an_atr():
    assert advice.trailing_stop(CFG, position(), flat_candles(n=5), 3000.0) is None


def test_under_a_plan_the_core_is_held_whatever_the_flags_say():
    a = run(position(), 3360, [sig("trend", -1)] + RISKY(3), plan_floor=1250.0)
    assert a.action == "HOLD" and a.suggested_fraction == 0.0 and a.suggested_stop is None
    assert any("adding plan" in r for r in a.reasons)
    warned = run(position(), 3360, BULL, plan_floor=1250.0, portfolio_flags=("LIQ_DANGER", "STRESS_BREACH"))
    assert warned.action == "HOLD" and any("liquidation price" in r for r in warned.reasons)


def test_under_a_plan_only_the_floor_ends_a_position():
    assert run(position(stop=1250.0), 1240, BULL, plan_floor=1250.0).action == "EXIT"
    assert run(position(), 3010, BULL, plan_floor=1250.0, portfolio_flags=("LIQUIDATED",)).action == "EXIT"
