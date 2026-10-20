from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from ethtracker import risk
from ethtracker.config import Config

CFG = Config()  # cross margin by default
ISOLATED = replace(CFG, margin_mode="isolated")
ISOLATED_NO_STRESS = replace(ISOLATED, stress_drop_pct=0)  # isolates the stop-vs-liquidation rule
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


def test_isolated_15x_blocks_a_stop_beyond_liquidation():
    plan = risk.plan_position(ISOLATED_NO_STRESS, 10_000, "long", 3000, 2760, 15)  # 8% stop, ~6.2% liquidation
    assert not plan.ok
    assert any("BEFORE your stop" in v for v in plan.violations)
    assert plan.max_leverage_for_stop < 15


def test_isolated_15x_boundary_between_3_and_3_2_percent_stops():
    assert risk.plan_position(ISOLATED_NO_STRESS, 10_000, "long", 3000, 2910, 15).ok  # 3.0% stop
    blocked = risk.plan_position(ISOLATED_NO_STRESS, 10_000, "long", 3000, 2904, 15)  # 3.2% stop
    assert not blocked.ok
    assert any("Max leverage for this stop" in v for v in blocked.violations)


def test_cross_margin_ignores_the_per_position_liquidation_distance():
    # the same 15x / 8% stop that isolated margin blocks: with a shared balance the position's own
    # isolated liquidation price is not what liquidates you, so the stop-vs-liquidation rule does not apply
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2760, 15)
    assert plan.ok
    assert not any("liquidat" in v.lower() for v in plan.violations)


def test_isolated_stress_demands_leverage_that_survives_the_move():
    too_high = risk.plan_position(ISOLATED, 10_000, "long", 3000, 2900, 10)
    assert any("Isolated margin" in v and "at most 2.5x" in v for v in too_high.violations)
    assert too_high.stress_loss_pct == pytest.approx(too_high.margin / 10_000)  # liquidated: loses its margin
    assert not too_high.stress_capped and too_high.stress_headroom_usd == float("inf")
    assert risk.plan_position(ISOLATED, 10_000, "long", 3000, 2900, 2).ok


def test_isolated_portfolio_flags_positions_whose_own_liquidation_is_inside_the_move():
    risky = [risk.mark_position(ISOLATED, make_position(leverage=10.0), 3000, NOW)]
    safe = [risk.mark_position(ISOLATED, make_position(leverage=2.0), 3000, NOW)]
    assert "STRESS_BREACH" in risk.portfolio_status(ISOLATED, 10_000, 10_000, risky).flags
    assert "STRESS_BREACH" not in risk.portfolio_status(ISOLATED, 10_000, 10_000, safe).flags


def test_cross_portfolio_flags_the_whole_book_liquidation_not_each_position():
    book = make_position(qty=8.0, leverage=10.0)  # whole-book liquidation ~1,758.8 on $10k
    danger = risk.portfolio_status(CFG, 10_000, 10_000, [risk.mark_position(CFG, book, 1790, NOW)])
    assert "LIQ_DANGER" in danger.flags and "LIQUIDATED" not in danger.flags
    gone = risk.portfolio_status(CFG, 10_000, 10_000, [risk.mark_position(CFG, book, 1750, NOW)])  # balance exhausted
    assert "LIQUIDATED" in gone.flags
    far = risk.portfolio_status(CFG, 10_000, 10_000, [risk.mark_position(CFG, book, 3000, NOW)])
    assert not {"LIQ_DANGER", "LIQUIDATED", "STRESS_BREACH"} & set(far.flags)
    assert far.account_liq_price == pytest.approx(1758.79, abs=0.01)


def test_max_leverage_for_distance_matches_the_stop_form():
    for side, stop in (("long", 2850), ("short", 3150)):
        by_stop = risk.max_leverage_for_stop(side, 3000, stop, 0.005, 0.5)
        assert risk.max_leverage_for_distance(side, 150 / 3000, 0.005, 0.5) == pytest.approx(by_stop)


def test_stop_inside_daily_noise_is_blocked_even_when_liquidation_is_safe():
    plan = risk.plan_position(ISOLATED, 10_000, "long", 3000, 2910, 10, atr_pct=0.04)  # 3% stop = 0.75x ATR
    assert plan.stop_fraction_of_liq <= CFG.max_stop_to_liq_fraction
    assert plan.stop_atr_multiple == pytest.approx(0.75)
    assert not plan.ok
    assert any("14d ATR" in v and "at most" in v for v in plan.violations)
    cross = risk.plan_position(CFG, 10_000, "long", 3000, 2910, 10, atr_pct=0.04)
    assert any("14d ATR" in v for v in cross.violations)
    assert not any("at most" in v for v in cross.violations)  # a leverage ceiling is an isolated-margin idea


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


NO_STRESS = replace(CFG, stress_drop_pct=0)
LIMIT_LONG = 10_000 / (0.40 + 0.005 * 0.60)  # equity / (d + m(1-d)): ~2.48x equity
LIMIT_SHORT = 10_000 / (0.40 + 0.005 * 1.40)


def test_stress_limit_is_about_2_5x_equity_and_asymmetric_for_shorts():
    assert risk.max_additional_notional(CFG, 10_000, 0, 0, "long") == pytest.approx(LIMIT_LONG)
    assert risk.max_additional_notional(CFG, 10_000, 0, 0, "short") == pytest.approx(LIMIT_SHORT)
    assert LIMIT_LONG / 10_000 == pytest.approx(2.48, abs=0.01)


def test_sizing_at_the_stress_limit_survives_but_loses_nearly_everything():
    plan = risk.plan_position(CFG, 10_000, "long", 3000, 2995, 5)  # tiny stop: risk sizing wants 12.5 ETH
    assert plan.stress_capped and not plan.margin_capped
    assert plan.notional == pytest.approx(LIMIT_LONG)
    assert plan.ok
    assert plan.stress_loss_pct == pytest.approx(0.40 * LIMIT_LONG / 10_000)  # ~99%
    assert plan.stress_loss_pct > 0.99
    assert any("not surviving financially" in w for w in plan.warnings)
    assert any("stress rule" in w for w in plan.warnings)


def test_stress_survival_flips_just_beyond_the_limit():
    m = CFG.maintenance_margin_rate
    inside = risk.stress_at(10_000, LIMIT_LONG, LIMIT_LONG, -0.40, m)
    outside = risk.stress_at(10_000, LIMIT_LONG * 1.01, LIMIT_LONG * 1.01, -0.40, m)
    assert inside.survives and not outside.survives
    assert inside.equity_after == pytest.approx(m * LIMIT_LONG * 0.60)


def test_explicit_qty_beyond_the_stress_limit_is_blocked():
    # a $5 stop keeps loss-at-stop inside the 1% budget so only the stress rule is in play
    plan = risk.plan_position(CFG, 100_000, "long", 3000, 2995, 5, qty=90.0)  # $270k notional vs ~$248k limit
    assert any("breach your margin limit" in v for v in plan.violations)
    assert risk.plan_position(CFG, 100_000, "long", 3000, 2995, 5, qty=80.0).ok  # $240k


def test_loss_cap_tightens_the_limit_and_enforces_it():
    capped = replace(CFG, max_stress_loss_pct=50)
    assert risk.max_additional_notional(capped, 10_000, 0, 0, "long") == pytest.approx(0.5 * 10_000 / 0.4)
    plan = risk.plan_position(capped, 10_000, "long", 3000, 2995, 5)
    assert plan.stress_capped and plan.stress_loss_pct == pytest.approx(0.5)
    assert not any("financially" in w for w in plan.warnings)
    blocked = risk.plan_position(capped, 10_000, "long", 3000, 2900, 5, qty=5.0)  # $15k: 60% loss
    assert any("would lose" in v for v in blocked.violations)


def test_headroom_shrinks_with_existing_holdings_marked_to_the_new_price():
    held = [make_position(entry=3000.0, qty=1.0)]
    total = risk.total_equity(10_000, held, 2400)
    net, gross = risk.exposure(held, 2400)
    assert (total, net, gross) == (9_400, 2_400, 2_400)
    expected = (9_400 - 0.40 * 2_400 - 0.005 * 0.60 * 2_400) / (0.40 + 0.005 * 0.60)
    assert risk.max_additional_notional(CFG, total, net, gross, "long") == pytest.approx(expected)
    plan = risk.plan_position(CFG, 10_000, "long", 2400, 2300, 5, holdings=held)
    assert plan.stress_headroom_usd == pytest.approx(expected)


def test_a_book_sized_to_the_limit_is_in_breach_after_a_further_fall():
    qty = LIMIT_LONG / 3000
    held = [make_position(entry=3000.0, qty=qty, leverage=5.0)]
    at_entry = risk.portfolio_status(CFG, 10_000, 10_000, [risk.mark_position(CFG, held[0], 3000, NOW)])
    assert at_entry.stress_survives and at_entry.headroom_notional == pytest.approx(0, abs=1e-6)
    after_fall = risk.portfolio_status(CFG, 10_000, 10_000, [risk.mark_position(CFG, held[0], 2400, NOW)])
    assert not after_fall.stress_survives and "STRESS_BREACH" in after_fall.flags
    assert after_fall.headroom_notional == 0


def test_account_liquidation_price_satisfies_the_margin_condition():
    m = 0.005
    long_book = [make_position(qty=8.0)]
    p = risk.account_liquidation_price(10_000, long_book, 3000, m)
    assert 10_000 + 8 * (p - 3000) == pytest.approx(m * 8 * p)
    short_book = [make_position(side="short", stop=3100.0, qty=8.0)]
    q = risk.account_liquidation_price(10_000, short_book, 3000, m)
    assert 10_000 + 8 * (3000 - q) == pytest.approx(m * 8 * q)
    assert q > 3000 > p
    assert risk.account_liquidation_price(10_000, long_book + short_book, 3000, m) is None
    assert risk.account_liquidation_price(10_000, [], 3000, m) is None


def test_a_hedged_book_survives_both_directions():
    book = [make_position(qty=5.0), make_position(id=2, side="short", stop=3100.0, qty=5.0)]
    net, gross = risk.exposure(book, 3000)
    assert net == 0 and gross == 30_000
    assert all(risk.stress_at(10_000, net, gross, x, 0.005).survives for x in (-0.4, 0.4))


def test_stress_rule_can_be_disabled():
    assert risk.max_additional_notional(NO_STRESS, 10_000, 0, 0, "long") == float("inf")
    plan = risk.plan_position(NO_STRESS, 10_000, "long", 3000, 2995, 5)
    assert plan.margin_capped and not plan.stress_capped
    assert not any("stress" in w.lower() for w in plan.warnings)
    held = [risk.mark_position(NO_STRESS, make_position(qty=50.0), 2000, NOW)]
    assert "STRESS_BREACH" not in risk.portfolio_status(NO_STRESS, 10_000, 10_000, held).flags


def test_margin_cap_shrinks_the_position_and_warns():
    plan = risk.plan_position(NO_STRESS, 10_000, "long", 3000, 2995, 5)
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


def test_mark_position_flags_stop_and_liquidation_danger_under_isolated_margin():
    status = risk.mark_position(ISOLATED, make_position(), 2750, NOW)  # liq is ~2713.6
    assert "LIQ_DANGER" in status.flags and "STOP_HIT" in status.flags
    assert status.unrealized == pytest.approx(-250)


def test_mark_position_reports_liquidated_past_the_liq_price_under_isolated_margin():
    assert "LIQUIDATED" in risk.mark_position(ISOLATED, make_position(), 2700, NOW).flags


def test_cross_margin_positions_carry_no_per_position_liquidation_flags():
    status = risk.mark_position(CFG, make_position(), 2700, NOW)  # beyond the isolated liq price
    assert status.flags == ("STOP_HIT",)
    assert status.dist_to_liq < 0  # the isolated estimate is still reported for reference


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
