from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ethtracker import risk
from ethtracker.config import Config

CFG = Config()
NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("lev", [2, 10, 15])
def test_equity_at_liquidation_equals_maintenance_margin(side, lev):
    entry, qty, mmr = 3000.0, 2.0, 0.005
    liq = risk.liquidation_price(side, entry, lev, mmr)
    equity = qty * entry / lev + risk.pnl(side, entry, liq, qty)
    assert equity == pytest.approx(mmr * qty * liq)


def test_liquidation_price_known_values():
    assert risk.liquidation_price("long", 1000, 10, 0.005) == pytest.approx(904.5226, rel=1e-6)
    assert risk.liquidation_price("short", 1000, 10, 0.005) == pytest.approx(1094.5274, rel=1e-6)


def test_liquidation_requires_leverage_of_at_least_one():
    with pytest.raises(ValueError):
        risk.liquidation_price("long", 1000, 0.5, 0.005)


@pytest.mark.parametrize("side,stop", [("long", 2850), ("short", 3150), ("long", 2400)])
def test_max_leverage_puts_stop_at_the_allowed_fraction_of_liquidation(side, stop):
    entry, mmr, fraction = 3000, 0.005, 0.5
    lev = risk.max_leverage_for_stop(side, entry, stop, mmr, fraction)
    liq = risk.liquidation_price(side, entry, lev, mmr)
    assert abs(entry - stop) / abs(entry - liq) == pytest.approx(fraction)


def test_loss_at_stop_equals_risk_budget_including_fees():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10)
    assert plan.risk_usd == pytest.approx(100.0)
    assert plan.risk_pct == pytest.approx(0.01)
    assert plan.qty == pytest.approx(100 / (100 + 0.0005 * (3000 + 2900)))
    assert plan.ok


def test_short_sizing_is_symmetric():
    plan = risk.plan_position(CFG, 10_000, "short", 3000, 3100, 10)
    assert plan.risk_usd == pytest.approx(100.0)
    assert plan.liq_price > plan.entry
    assert plan.ok


def test_position_size_does_not_depend_on_leverage_when_uncapped():
    low = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 5)
    high = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10)
    assert low.qty == pytest.approx(high.qty)
    assert low.margin == pytest.approx(2 * high.margin)


def test_15x_blocks_a_stop_beyond_liquidation():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2760, 15)  # 8% stop, ~6.2% liquidation
    assert not plan.ok
    assert "BEFORE your stop" in plan.violations[0]
    assert plan.max_leverage_for_stop < 15


def test_15x_boundary_between_3_and_3_2_percent_stops():
    assert risk.plan_position(CFG, 10_000, "long", 3000, 2910, 15).ok  # 3.0% stop
    blocked = risk.plan_position(CFG, 10_000, "long", 3000, 2904, 15)  # 3.2% stop
    assert not blocked.ok
    assert "Max leverage for this stop" in blocked.violations[0]


def test_max_leverage_for_distance_matches_the_stop_form():
    for side, stop in (("long", 2850), ("short", 3150)):
        by_stop = risk.max_leverage_for_stop(side, 3000, stop, 0.005, 0.5)
        assert risk.max_leverage_for_distance(side, 150 / 3000, 0.005, 0.5) == pytest.approx(by_stop)


def test_stop_inside_daily_noise_is_blocked_even_when_liquidation_is_safe():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2910, 10, atr_pct=0.04)  # 3% stop = 0.75x ATR
    assert plan.stop_fraction_of_liq <= CFG.max_stop_to_liq_fraction
    assert plan.stop_atr_multiple == pytest.approx(0.75)
    assert not plan.ok
    assert "14d ATR" in plan.violations[0]
    assert "at most" in plan.violations[0]


def test_stop_beyond_the_atr_minimum_passes_at_low_leverage():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2730, 3, atr_pct=0.04)  # 9% stop = 2.25x ATR
    assert plan.ok
    assert plan.stop_atr_multiple == pytest.approx(2.25)


def test_atr_rule_warns_when_atr_is_unknown_and_can_be_disabled():
    unknown = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10)
    assert unknown.ok and any("ATR not available" in w for w in unknown.warnings)
    off = risk.plan_position(replace(CFG, min_stop_atr_multiple=0), 10_000, "long", 3000, 2910, 10, atr_pct=0.04)
    assert off.ok and not any("ATR" in w for w in off.warnings)


def test_leverage_above_cap_is_a_violation():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2970, 20)
    assert any("exceeds" in v for v in plan.violations)


def test_margin_cap_shrinks_the_position_and_warns():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2995, 5)
    assert plan.margin_capped
    assert plan.margin == pytest.approx(5_000)
    assert plan.risk_pct < 0.01
    assert plan.ok
    assert any("capped" in w for w in plan.warnings)


def test_open_risk_cap_counts_existing_positions():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10, open_risk_usd=250)
    assert any("Total open risk" in v for v in plan.violations)


def test_explicit_qty_larger_than_budget_is_blocked():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10, qty=3.0)
    assert any("budget" in v for v in plan.violations)


def test_reward_risk_is_net_of_fees_and_funding_and_warns_when_thin():
    good = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10, target=3400)
    thin = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 10, target=3150)
    gross = good.qty * 400
    expected = gross - 0.0005 * good.qty * (3000 + 3400) - good.funding_usd
    assert good.reward_usd == pytest.approx(expected)
    assert good.reward_risk > CFG.min_reward_risk
    assert any("reward:risk" in w for w in thin.warnings)


def test_funding_sign_and_size():
    assert risk.funding_cost("long", 1000, 0.0001, 30) == pytest.approx(9.0)
    assert risk.funding_cost("short", 1000, 0.0001, 30) == pytest.approx(-9.0)


@pytest.mark.parametrize(
    "side,entry,stop,target",
    [("long", 3000, 3100, None), ("short", 3000, 2900, None), ("long", 3000, 2900, 2800), ("short", 3000, 3100, 3200)],
)
def test_levels_on_the_wrong_side_are_rejected(side, entry, stop, target):
    with pytest.raises(ValueError):
        risk.plan_position(CFG, 10_000, side, entry, stop, 10, target=target)


def test_scenarios_cap_loss_at_margin_once_liquidated():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2900, 12)
    by_move = {round(s.move, 2): s for s in risk.scenarios(plan, 10_000)}
    assert by_move[-0.10].liquidated
    assert by_move[-0.10].pnl == pytest.approx(-plan.margin)
    assert not by_move[-0.05].liquidated
    assert by_move[-0.05].pnl == pytest.approx(-0.05 * plan.notional)
    assert by_move[0.10].pnl == pytest.approx(0.10 * plan.notional)


def make_position(**overrides) -> risk.Position:
    fields = dict(
        id=1, side="long", entry=3000.0, qty=1.0, leverage=10.0, stop=2900.0, target=None,
        opened_at=(NOW - timedelta(days=10)).isoformat(), mmr=0.005, initial_risk_usd=100.0,
    )
    fields.update(overrides)
    return risk.Position(**fields)


def test_mark_position_flags_stop_and_liquidation_danger():
    status = risk.mark_position(CFG, make_position(), 2750, NOW)  # liq is ~2713.6
    assert "LIQ_DANGER" in status.flags and "STOP_HIT" in status.flags
    assert status.unrealized == pytest.approx(-250)


def test_mark_position_reports_liquidated_past_the_liq_price():
    assert "LIQUIDATED" in risk.mark_position(CFG, make_position(), 2700, NOW).flags


def test_mark_position_time_flags():
    assert CFG.max_holding_days == 14  # a two-week swing
    warn = risk.mark_position(CFG, make_position(opened_at=(NOW - timedelta(days=12)).isoformat()), 3000, NOW)
    stop = risk.mark_position(CFG, make_position(opened_at=(NOW - timedelta(days=15)).isoformat()), 3000, NOW)
    assert warn.flags == ("TIME_WARN",)
    assert "TIME_STOP" in stop.flags


def test_loss_to_stop_includes_exit_fee_and_floors_at_zero():
    above = risk.mark_position(CFG, make_position(), 3100, NOW)
    assert above.loss_to_stop_usd == pytest.approx(200 + 0.0005 * 2900)
    below = risk.mark_position(CFG, make_position(), 2800, NOW)
    assert below.loss_to_stop_usd == pytest.approx(0.0005 * 2900)


def test_short_position_distances_flip_sign():
    short = make_position(side="short", stop=3100.0)
    status = risk.mark_position(CFG, short, 2900, NOW)
    assert status.unrealized == pytest.approx(100)
    assert status.dist_to_stop == pytest.approx((3100 - 2900) / 2900)
    assert status.liq_price > 3000


def test_portfolio_flags_open_risk_and_drawdown():
    statuses = [risk.mark_position(CFG, make_position(id=i), 3100, NOW) for i in (1, 2)]
    port = risk.portfolio_status(CFG, 10_000, 10_000, statuses)
    assert port.unrealized == pytest.approx(200)
    assert port.gross_notional == pytest.approx(6200)
    assert "OPEN_RISK_LIMIT" in port.flags  # ~$403 / $10,200 = 3.95% > 3%
    deep = risk.portfolio_status(CFG, 10_000, 13_000, statuses[:1])
    assert "DRAWDOWN_LIMIT" in deep.flags
    ok = risk.portfolio_status(CFG, 10_000, 10_000, statuses[:1])
    assert ok.flags == ()
