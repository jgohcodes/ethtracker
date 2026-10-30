"""Command-line interface: ``ethtracker <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import advice, alerts, coinbase, combo, config, cycles, decisions, derivs, dipbuy, indicators, observed, plan, score, pullbacks, risk, shorts, signals, snapshot, study, swing
from .data import DataError, Provider, get_provider
from .journal import Journal
from .store import CachedProvider


class CliError(Exception):
    pass


@dataclass
class Ctx:
    cfg: config.Config
    journal: Journal
    home: Path


def usd(x: float, digits: int = 2) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.{digits}f}"


def signed_usd(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}${abs(x):,.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def na(x, fmt) -> str:
    return "n/a" if x is None else fmt(x)


def _provider(args, ctx: Ctx, product: str = "ETH-USD") -> Provider:
    """The configured source; network sources go through the local candle cache unless --no-cache."""
    provider = get_provider(args.provider or ctx.cfg.provider, args.csv, product)
    if provider.name == "csv" or args.no_cache:
        return provider
    return CachedProvider(provider, ctx.home / "candles")


def _equity(args, ctx: Ctx) -> float:
    equity = getattr(args, "equity", None)
    equity = ctx.journal.equity if equity is None else equity
    if equity is None:
        raise CliError("equity is not set: run `ethtracker equity <USD>` or pass --equity")
    return equity


def _open_risk(ctx: Ctx, price: float, now: datetime) -> float:
    return sum(risk.mark_position(ctx.cfg, p, price, now).loss_to_stop_usd for p in ctx.journal.open_positions())


def history_lines(df, side: str, horizon: int, stop_dist: float | None = None, liq_dist: float | None = None) -> list[str]:
    mae = indicators.adverse_excursions(df, side, horizon)
    if len(mae) < 30:
        return [f"History: only {len(mae)} rolling {horizon}d windows available; skipped (need more candles)."]
    q50, q75, q90 = np.quantile(mae, [0.5, 0.75, 0.9])
    lines = [
        f"Worst move against a {side} within {horizon}d of entry, {len(mae)} rolling windows "
        f"({df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}): median {pct(q50)}, p75 {pct(q75)}, p90 {pct(q90)}"
    ]
    if stop_dist is not None:
        lines.append(f"  your stop ({pct(stop_dist)}) would have been touched in {pct(float((mae >= stop_dist).mean()), 0)} of windows")
    if liq_dist is not None:
        lines.append(f"  your liquidation ({pct(liq_dist)}) would have been touched in {pct(float((mae >= liq_dist).mean()), 0)} of windows")
    lines.append("  (windows overlap heavily: a rough scale, not a probability)")
    return lines


def print_plan(plan: risk.SizePlan, equity: float, cross: bool) -> None:
    arrow = "LONG" if plan.side == "long" else "SHORT"
    target = f"  target {usd(plan.target)}" if plan.target else ""
    print(f"ETHUSD {arrow} @ {plan.leverage:g}x   entry {usd(plan.entry)}  stop {usd(plan.stop)}{target}")
    print("-" * 78)
    print(f"Size         {plan.qty:.4f} ETH   notional {usd(plan.notional)}   margin {usd(plan.margin)} ({pct(plan.margin_pct_equity)} of equity)")
    print(f"Exposure     {plan.effective_leverage:.2f}x equity (notional / equity), equity {usd(equity)}")
    print(f"Risk         loss at stop {usd(plan.risk_usd)} = {pct(plan.risk_pct, 2)} of equity (fees {usd(plan.fees_at_stop_usd)} included)")
    if cross:
        print(f"Liquidation  cross margin: set by the whole book, not this position (see Stress). Isolated-margin equivalent at {plan.leverage:g}x: {usd(plan.liq_price)}")
    else:
        print(f"Liquidation  {usd(plan.liq_price)} ({pct(plan.liq_dist)} from entry); stop is {pct(plan.stop_dist)} away = {pct(plan.stop_fraction_of_liq, 0)} of that distance")
        print(f"Max leverage for this stop: {plan.max_leverage_for_stop:.1f}x")
    if plan.stress_move:
        fall = "fall" if plan.side == "long" else "rise"
        room = f"; room before it: {usd(plan.stress_headroom_usd, 0)} notional" if cross else ""
        capped = "  [size capped by this rule]" if plan.stress_capped else ""
        print(f"Stress       a {abs(plan.stress_move):.0%} {fall} loses {pct(plan.stress_loss_pct, 0)} of equity with this trade added{room}{capped}")
    if plan.stop_atr_multiple is not None:
        print(f"Stop width   {plan.stop_atr_multiple:.1f}x the 14d ATR")
    direction = "paid" if plan.funding_usd >= 0 else "received"
    print(f"Funding      ~{usd(abs(plan.funding_usd))} {direction} over {plan.hold_days:g}d (estimate; the rate floats)")
    if plan.reward_usd is not None:
        print(f"Reward       {usd(plan.reward_usd)} net of fees and funding -> {plan.reward_risk:.2f}R")


def print_verdict(plan: risk.SizePlan) -> None:
    for w in plan.warnings:
        print(f"  warning: {w}")
    if plan.ok:
        print("VERDICT: PASS")
    else:
        print("VERDICT: BLOCKED")
        for v in plan.violations:
            print(f"  - {v}")


def print_ladder(cfg: config.Config, equity: float, args, entry: float, open_risk: float, holdings) -> None:
    print("Leverage ladder for this stop (same risk budget; only margin and liquidation move; liquidation rules only):")
    print(f"  {'lev':>5} {'margin':>12} {'liq price':>12} {'liq dist':>9} {'stop/liq':>9}  verdict")
    for lev in (3, 5, 7, 10, 12, 15):
        p = risk.plan_position(cfg, equity, args.side, entry, args.stop, lev, target=args.target,
                               risk_pct=args.risk_pct, hold_days=args.hold_days, open_risk_usd=open_risk,
                               holdings=holdings)
        marker = "ok" if p.ok else "blocked"
        print(f"  {lev:>4}x {usd(p.margin):>12} {usd(p.liq_price):>12} {pct(p.liq_dist):>9} {pct(p.stop_fraction_of_liq, 0):>9}  {marker}")


def _fetch_candles(provider: Provider | None, days: int):
    """Candles, or None when the download fails: a missing history weakens a check but must not block the command."""
    try:
        return provider.candles(days)
    except DataError as exc:
        print(f"warning: candles unavailable ({exc})", file=sys.stderr)
        return None


def _atr_pct(df) -> float | None:
    return None if df is None else indicators.snapshot(df).atr_14_pct


def cmd_init(args, ctx: Ctx) -> int:
    path = ctx.home / "config.toml"
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
    else:
        ctx.home.mkdir(parents=True, exist_ok=True)
        path.write_text(config.render_default_toml())
        print(f"wrote {path}")
    if args.equity is not None:
        ctx.journal.set_equity(args.equity)
        ctx.journal.save()
        print(f"equity set to {usd(args.equity)}")
    return 0


def cmd_equity(args, ctx: Ctx) -> int:
    if args.value is not None:
        ctx.journal.set_equity(args.value)
        ctx.journal.save()
    j = ctx.journal
    if j.equity is None:
        print("equity not set")
    else:
        print(f"equity {usd(j.equity)}   recorded peak {usd(j.peak_equity)}")
    return 0


def _long_book(ctx: Ctx, price: float) -> tuple[float, float, float | None]:
    """Total equity (wallet plus open PnL), ETH held long, and the account's liquidation price, at ``price``."""
    now = datetime.now(timezone.utc)
    longs = [p for p in ctx.journal.open_positions() if p.side == "long"]
    statuses = [risk.mark_position(ctx.cfg, p, price, now) for p in longs]
    total = (ctx.journal.equity or 0.0) + sum(s.unrealized for s in statuses)
    qty = sum(p.qty for p in longs)
    liq = risk.account_liquidation_price(total, longs, price, ctx.cfg.maintenance_margin_rate) if longs and total > 0 else None
    return total, qty, liq


def _print_plan(ctx: Ctx, price: float) -> None:
    total, qty, liq = _long_book(ctx, price)
    s = plan.sizing(ctx.cfg, ctx.journal.plan, price, total, qty, liq)
    print(f"Plan: liquidation never above {usd(s['floor'])}; add {usd(s['weekly_add'])} a week. Price {usd(price)}. "
          f"Maintenance margin {ctx.cfg.maintenance_margin_rate:.1%}.")
    print(f"Book: equity {usd(total)}, {qty:.4f} ETH long, liquidation "
          + (usd(liq) if liq is not None else "none") + f" ({s['status']}).")
    if s["cap_leverage"] is None:
        print("Price is at or under the floor: nothing can be added without lifting the liquidation price above it.")
        return
    print(f"New cash can carry up to {s['cap_leverage']:.2f}x here. This week's {usd(s['weekly_add'])} buys up to "
          f"{s['weekly_qty']:.4f} ETH ({usd(s['weekly_notional'])}). Room without new cash: {s['room_qty']:.4f} ETH.")
    if s["status"] == "above the floor":
        print(f"Liquidation is above the floor: cut {s['cut_qty']:.4f} ETH, or add {usd(s['margin_needed'])} of margin.")


def cmd_plan(args, ctx: Ctx) -> int:
    j = ctx.journal
    if args.floor is not None or args.weekly is not None or args.mmr is not None:
        current = j.plan or {}
        floor = args.floor if args.floor is not None else current.get("floor")
        weekly = args.weekly if args.weekly is not None else current.get("weekly_add")
        if floor is None or weekly is None:
            raise CliError("the first plan needs both --floor and --weekly")
        mmr = args.mmr if args.mmr is not None else current.get("mmr", plan.PLAN_MMR)
        j.set_plan(floor, weekly, mmr)
        j.save()
        ctx = replace(ctx, cfg=plan.with_plan_margin(ctx.cfg, j.plan))
    if not j.plan:
        print("No plan set: ethtracker plan --floor 1250 --weekly 500")
        return 0
    _print_plan(ctx, args.price if args.price is not None else _provider(args, ctx).spot())
    return 0


def cmd_add(args, ctx: Ctx) -> int:
    """Record new cash and the ETH it bought, sized so the liquidation price stays at or below the plan's floor."""
    j = ctx.journal
    if not j.plan:
        raise CliError("set a plan first: ethtracker plan --floor 1250 --weekly 500")
    try:
        at = decisions.timestamp(args.at)
    except ValueError as exc:
        raise CliError(f"--at: not a date or time: {args.at!r}") from exc
    if at > pd.Timestamp(datetime.now(timezone.utc)) + pd.Timedelta(minutes=5):
        raise CliError("--at is in the future")
    if args.at and args.price is None:
        raise CliError("--at needs --price: the price you paid then")
    cash = j.plan["weekly_add"] if args.usd is None else args.usd
    price = args.price if args.price is not None else _provider(args, ctx).spot()
    floor, mmr, fee = j.plan["floor"], ctx.cfg.maintenance_margin_rate, ctx.cfg.taker_fee_rate
    span = price - floor * (1 - mmr)
    if span <= 0:
        raise CliError(f"price {usd(price)} is at or under the floor {usd(floor)}: nothing can be added")
    total, qty, _ = _long_book(ctx, price)
    allowed = max(0.0, (total + cash - qty * span) / (span + fee * price))  # the entry fee comes out of equity too
    buy = allowed if args.qty is None else args.qty
    if buy <= 0:
        raise CliError("no room to add: the liquidation price is already at the floor")
    if buy > allowed * (1 + 1e-9) and not args.force:
        raise CliError(f"{buy:.4f} ETH would lift the liquidation price above {usd(floor)}; the most is {allowed:.4f} "
                       "(use --force to record it anyway)")
    if cash:
        j.deposit(cash)
    pos = j.open(opened_at=at.to_pydatetime().isoformat(timespec="seconds"), side="long", entry=price, qty=buy,
                 leverage=buy * price / cash if cash else price / span, stop=floor, target=None, mmr=mmr,
                 initial_risk_usd=buy * (price - floor), note=args.note or "plan add")
    j.save()
    print(f"Recorded #{pos.id}: {usd(cash)} added, {buy:.4f} ETH bought at {usd(price)} ({usd(buy * price)}).")
    record = _capture(args, ctx, at, price, floor)
    j.record_decision(pos.id, {"usd": cash, "qty": buy, **record})
    j.save()
    _print_decision(record)
    _print_plan(ctx, price)
    print("Run `ethtracker publish --once` to update the dashboard.")
    return 0


def _funding_until(ctx: Ctx, at: pd.Timestamp) -> pd.Series | None:
    """About a year of daily funding up to ``at``: the cached year for a buy made now, else fetched for that date."""
    if pd.Timestamp(datetime.now(timezone.utc)) - at < pd.Timedelta(days=2):
        return _recent_funding(ctx, derivs.HISTORY_INSTRUMENT)
    try:
        series, _ = derivs.fetch_funding_history(at - pd.Timedelta(days=366), at, instrument=derivs.HISTORY_INSTRUMENT)
    except DataError as exc:
        print(f"warning: funding history unavailable ({exc})", file=sys.stderr)
        return None
    return series


def _capture(args, ctx: Ctx, at: pd.Timestamp, price: float, floor: float | None,
             history: pd.DataFrame | None = None) -> dict:
    """What the signals said at ``at``; if they cannot be read, why, so `review --backfill` can fill it later."""
    try:
        if history is None:
            history, _ = cycles.ignore_flash_prints(_provider(args, ctx).candles(CONTEXT_DAYS))
        return decisions.capture(ctx.cfg, history, at, price, _funding_until(ctx, at), floor)
    except (DataError, ValueError) as exc:
        print(f"warning: the signals at this buy were not captured ({exc}); run `ethtracker review --backfill` later",
              file=sys.stderr)
        return {"at": at.isoformat(), "price": price, "error": str(exc)}


STANCE_WORDS = {"with": "WITH the signals", "mixed": "MIXED signals", "against": "AGAINST the signals"}


def _print_decision(record: dict) -> None:
    if "error" in record:
        return
    parts = ", ".join(f"{k.replace('_', ' ')} {v:+d}" for k, v in record["parts"].items() if v) or "nothing either way"
    votes = ", ".join(f"{k} {v:+d}" for k, v in record["votes"].items() if v) or "none"
    print(f"Signals on the {record['basis_close']} close: {STANCE_WORDS[record['stance']]} (score {record['score']:+d}: {parts}).")
    print(f"  Trend {record['trend']}; family votes {votes}; risk flags {record['risk_flags']}; crowding "
          f"{record['crowding'] or 'unknown'}; pullback-over signals in the last 5 days: "
          f"{', '.join(record['pullback_signals']) or 'none'}.")
    print(f"  Stored with the buy, timestamped {record['at']}; see `ethtracker review` and the dashboard.")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def cmd_review(args, ctx: Ctx) -> int:
    """Your buys against what the signals said when you made them, and how each turned out."""
    j = ctx.journal
    provider = _provider(args, ctx)
    history, _ = cycles.ignore_flash_prints(provider.candles(CONTEXT_DAYS))
    if args.backfill:
        have = {d["position_id"] for d in j.decisions if "error" not in d}
        todo = [p for p in j.positions if p.side == "long" and p.id not in have]
        for p in todo:
            old = next((d for d in j.decisions if d.get("position_id") == p.id), {})
            record = _capture(args, ctx, decisions.timestamp(p.opened_at), p.entry, j.plan["floor"] if j.plan else None, history)
            j.record_decision(p.id, {"usd": old.get("usd"), "qty": p.qty, **record})
        j.save()
        print(f"Captured the signals for {len(todo)} buy(s) without a record, from the close before each buy.")
    try:
        price = provider.spot()
    except DataError:
        price = None
    block = decisions.review(snapshot.closed_candles(history, datetime.now(timezone.utc)), j.decisions,
                             {p.id: p for p in j.positions}, price)
    if block is None:
        print("No buys with captured signals yet: `ethtracker add` records them" +
              (f" ({sum(1 for d in j.decisions if 'error' in d)} failed: run `ethtracker review --backfill`)" if j.decisions else ""))
        return 0
    print(f"Your buys against the signals: judged on {decisions.JUDGE_DAYS} days (on {decisions.EARLY_DAYS} while newer); "
          "signals as they read on the close before each buy")
    print(f"  {'#':>3} {'date':<10} {'price':>10} {'ETH':>8} {'signals':<8} {'score':>5} {'30d':>7} {'90d':>7} {'now':>7} "
          f"{'low 30d':>8} {'you':>5} {'signals':>8}")
    verdict = {True: "right", False: "wrong", None: "-"}
    for r in block["rows"]:
        o = r["outcome"]
        print(f"  {r['position_id']:>3} {r['at'][:10]:<10} {r['price']:>10,.2f} {r.get('qty') or 0:>8.4f} {r['stance']:<8} "
              f"{r['score']:>+5d} {_pct(o['returns'][30]):>7} {_pct(o['returns'][90]):>7} {_pct(o['now']):>7} "
              f"{_pct(o['low_30']):>8} {verdict[o['right']]:>5} {verdict[o['signals_right']]:>8}")
    s = block["summary"]
    print()
    for stance in ("with", "mixed", "against"):
        g = s["by_stance"][stance]
        if g["buys"]:
            print(f"  {STANCE_WORDS[stance]}: {g['buys']} buy(s); judged {g['judged']}"
                  + (f", right {g['right']:.0%}, median {g['median_return']:+.1%}" if g["judged"] else ""))
    if s["against_judged"]:
        print(f"  You went against the signals and were right {s['against_and_right']} of {s['against_judged']} time(s).")
    missed = [(r, m) for r in block["rows"] for m in r["outcome"]["misses"]]
    unflagged = [(r, u) for r in block["rows"] for u in r["outcome"]["unflagged"]]
    if missed or unflagged:
        print()
        print("What the signals got wrong, 30 days after each buy:")
        for r, u in unflagged:
            print(f"  #{r['position_id']} {r['at'][:10]}: {u}")
        for r, m in missed:
            print(f"  #{r['position_id']} {r['at'][:10]}: {m['headline']} (price then went {_pct(r['outcome']['returns'][30])})")
    if s["signal_hits"]:
        print()
        print("Each signal on your buys: how often its direction matched the next 30 days")
        for h in s["signal_hits"]:
            print(f"  {h['name']:<20} {h['calls']:>4} call(s)  right {h['right']:.0%}")
    if block["missing"]:
        print(f"\n{block['missing']} buy(s) have no captured signals: run `ethtracker review --backfill`.")
    return 0


CONTEXT_FLOORS = (0.30, 0.40, 0.50, 0.60)  # under the last close: generic, so the output is safe for public logs


def cmd_context(args, ctx: Ctx) -> int:
    """The dashboard's market panels in the terminal: trend, crowding, the pullback signals' record, buy-order
    fill rates and the risk of reaching a floor. Market facts only, unless you pass your own --floor."""
    provider = _provider(args, ctx)
    market = _market_context(ctx, provider)
    full, _ = cycles.ignore_flash_prints(snapshot.closed_candles(market["history"], datetime.now(timezone.utc)))
    last = float(full["close"].iloc[-1])
    t = plan.trend(full)
    print(f"ETH on the {full.index[-1]:%Y-%m-%d} close: {usd(last)} ({len(full)} daily candles from {full.index[0]:%Y-%m-%d})")
    print(f"Trend: {t['label']}. 50-day {na(t['sma50'], usd)} ({'above' if t['above_50'] else 'below'}), "
          f"200-day {na(t['sma200'], usd)} ({'above' if t['above_200'] else 'below'}), 28 days {na(t['return_28d'], lambda x: f'{x:+.1%}')}")
    c = plan.crowding(market["funding8h"])
    print("Crowding: funding history unavailable" if c is None else
          f"Crowding: {c['label']}. Funding {c['funding_8h']:.4%}/8h, at the {c['percentile']:.0%} point of the last "
          f"{c['days']} days (median {c['median_8h']:.4%}, high {c['high_8h']:.4%})")

    floors = [last * (1 - d) for d in CONTEXT_FLOORS] + ([args.floor] if args.floor else [])
    risks = [plan.floor_risk(full, last, f) for f in floors]
    vol = risks[0]["vol_forecast"]
    print(f"Volatility: EWMA forecast {na(vol, pct)} a year, 30-day realised {na(risks[0]['vol_30d'], pct)}")
    hist = next((r["history"] for r in risks if r["history"]), None)
    print()
    print("Reaching a floor. Touch: a driftless random walk at the forecast volatility. History: from days "
          + (f"{hist['from_depth']:.0%}+ under the high (like today), {hist['starts']} starts in {hist['drawdowns']} drawdowns"
             if hist else "as deep as today (none with four years after them)"))
    print(f"  {'floor':>10} {'under':>7} {'sigmas':>7} {'30 days':>8} {'1 year':>7} {'hist 1y':>8} {'hist 4y':>8}")
    for f, r in zip(floors, risks):
        h = r["history"] or {}
        print(f"  {usd(f, 0):>10} {1 - f / last:>7.0%} {na(r['sigmas'], lambda x: f'{x:.2f}'):>7} "
              f"{na(r['touch_30d'], lambda x: f'{x:.1%}'):>8} {na(r['touch_1y'], lambda x: f'{x:.0%}'):>7} "
              f"{na(h.get('within_1y'), lambda x: f'{x:.0%}'):>8} {na(h.get('within_4y'), lambda x: f'{x:.0%}'):>8}")

    o = plan.odds(full)
    print()
    print(f"Pullback signals ({'in' if o['in_pullback'] else 'not in'} a {o['pullback_depth']:.0%} pullback now). "
          f"Low held: no further {o['further_fall']:.0%} fall within {o['held_days']} days of the close")
    print(f"  {'signal':<40} {'last fired':>12} {'times':>6} {'low held':>9} {'95% low':>8} {'up 1y':>6}")
    for r in o["signals"]:
        fired = "baseline" if r["baseline"] else (r["last_fired"] or "never")
        print(f"  {r['name']:<40} {fired:>12} {r['events']:>6} {na(r['held'], lambda x: f'{x:.0%}'):>9} "
              f"{na(r['held_low'], lambda x: f'{x:.0%}'):>8} {na(r['higher_365'], lambda x: f'{x:.0%}'):>6}")

    print()
    print("A buy order under the close, good for 30 days, placed on every "
          + ("pullback day (like today)" if o["in_pullback"] else "day"))
    print(f"  {'under':>6} {'filled':>7} {'low held':>9} {'up 1y':>6}")
    for depth, rung in sorted(plan.ladder_stats(full).items()):
        print(f"  {depth:>6.0%} {na(rung.filled, lambda x: f'{x:.0%}'):>7} {na(rung.held, lambda x: f'{x:.0%}'):>9} "
              f"{na(rung.higher_365, lambda x: f'{x:.0%}'):>6}")
    return 0


def _candidate_states(cfg, full: pd.DataFrame) -> pd.DataFrame:
    """Signals left out of the score, as daily states by the dashboard's own rules, to show why they are out."""
    tech = signals.technical_series(cfg, full)
    rsi = np.where(tech["rsi"] >= cfg.rsi_overbought, -1.0, np.where(tech["rsi"] <= cfg.rsi_oversold, 1.0, 0.0))
    ext = np.where(tech["z20"] >= cfg.extension_z, -1.0, np.where(tech["z20"] <= -cfg.extension_z, 1.0, 0.0))
    return pd.DataFrame({"rsi (not scored)": np.where(tech["rsi"].isna(), np.nan, rsi),
                         "extension (not scored)": np.where(tech["z20"].isna(), np.nan, ext)}, index=full.index)


SCORE_ERAS = (("all", None, None), ("to 2020", None, "2021-01-01"), ("2021 on", "2021-01-01", None))


def _setups(full: pd.DataFrame) -> dict[str, pd.Series]:
    """Entry setups traders cite, as days they fire, each from data up to that day's close only."""
    close = full["close"]
    rsi = indicators.rsi(close)
    s10, s50 = indicators.sma(close, 10), indicators.sma(close, 50)
    rising, falling = s50 > s50.shift(10), s50 < s50.shift(10)
    # bullish divergence: today's close is below the lowest close of 5 to 30 days ago, but RSI is above RSI on that day
    prior_low = close.shift(5).rolling(26).min()
    prior_day = close.shift(5).rolling(26).apply(lambda w: w.argmin(), raw=True)
    pos = np.arange(len(close))
    idx = (pos - 5 - (25 - prior_day.fillna(0).astype(int))).clip(0)
    rsi_then = pd.Series(rsi.to_numpy()[idx], index=close.index).where(prior_low.notna())
    divergence = (close < prior_low) & (rsi > rsi_then) & (rsi < 50)
    recent_div = divergence.rolling(5, min_periods=1).max().astype(bool)
    return {
        "Every day (baseline)": close.notna(),
        "50d rising, RSI <= 40": rising & (rsi <= 40),
        "50d falling, RSI <= 30": falling & (rsi <= 30),
        "Bullish RSI divergence": divergence,
        "Divergence, then close > 10d avg": recent_div & (close > s10) & ~(close.shift(1) > s10.shift(1)),
    }


def cmd_score_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(args, ctx, "the signal score: does each part point the right way, "
                                                            "and what followed each score?")
    st = score.states(cfg, full, funding8h)
    fwd = score.forward(full, args.horizon)
    every = pd.concat([st, _candidate_states(cfg, full)], axis=1)

    def era(mask_index, start, end):
        keep = pd.Series(True, index=mask_index)
        if start:
            keep &= mask_index >= pd.Timestamp(start, tz="UTC")
        if end:
            keep &= mask_index < pd.Timestamp(end, tz="UTC")
        return keep

    print()
    print(f"Each part against ETH's next {args.horizon} days. IC: rank correlation of the state with the forward return "
          "(positive = points the right way). Overlapping windows: about one independent reading a month.")
    print(f"  {'part':<26} {'weight':>6} " + " ".join(f"{name:>22}" for name, *_ in SCORE_ERAS))
    print(f"  {'':<26} {'':>6} " + " ".join(f"{'IC  up-when+  up-when-':>22}" for _ in SCORE_ERAS))
    for col in every.columns:
        cells = []
        for _, start, end in SCORE_ERAS:
            m = era(full.index, start, end) & every[col].notna() & fwd["ret"].notna()
            x, y = every.loc[m, col], fwd.loc[m, "ret"]
            ic = score.rank_ic(x, y)
            up_pos = (y[x > 0] > 0).mean() if (x > 0).any() else float("nan")
            up_neg = (y[x < 0] > 0).mean() if (x < 0).any() else float("nan")
            cells.append(f"{ic:+.2f}  {up_pos:>6.0%}  {up_neg:>7.0%}" if np.isfinite(ic) else f"{'n/a':>22}")
        print(f"  {col:<26} {score.WEIGHTS.get(col, 0):>6.3f} " + " ".join(f"{c:>22}" for c in cells))

    noise = score.chance_ic(cfg, full, horizon=args.horizon)
    print(f"  Chance alone: with ETH's own days shuffled, the score's IC spreads about +/-{noise:.2f} (one sd), so an IC "
          f"beyond about +/-{2 * noise:.2f} is unlikely to be luck.")

    variants = {
        "literature weights (the score)": score.WEIGHTS,
        "equal weights, all five": {k: 1.0 for k in score.WEIGHTS},
        "trend only (three parts)": {"momentum": 1.0, "trend": 1.0, "long_trend": 1.0},
    }
    print()
    print("Whole scores against the next {} days: IC by era".format(args.horizon))
    for name, w in variants.items():
        sc = score.combine(st, w)
        ics = []
        for _, start, end in SCORE_ERAS:
            m = era(full.index, start, end) & sc.notna() & fwd["ret"].notna()
            ics.append(score.rank_ic(sc[m], fwd.loc[m, "ret"]))
        print(f"  {name:<34} " + "  ".join(f"{e}: {ic:+.2f}" for (e, *_), ic in zip(SCORE_ERAS, ics)))

    fine = score.combine(score.continuous(full, funding8h))
    step = score.combine(st)
    print()
    print(f"Continuous score (each part on a sliding scale) against the step score, next {args.horizon} days")
    for name, s in (("step score (shown)", step), ("continuous score", fine)):
        ics = []
        for _, start, end in SCORE_ERAS:
            m = era(full.index, start, end) & s.notna() & fwd["ret"].notna()
            ics.append(score.rank_ic(s[m], fwd.loc[m, "ret"]))
        flips = score.flips_per_year(s)
        moves = s.diff().abs()
        print(f"  {name:<22} " + "  ".join(f"{e}: {ic:+.2f}" for (e, *_), ic in zip(SCORE_ERAS, ics))
              + f"  | days it moved: {(moves > 1e-9).mean():.0%}, sign flips a year: {flips:.0f}")
    both = step.notna() & fine.notna()
    print(f"  correlation between the two: {score.rank_ic(step[both], fine[both]):+.2f}")

    print()
    print(f"Entry setups: what followed the days each fired, {args.horizon} days on (higher / median / fell 10%+ within)")
    print(f"  {'setup':<36} " + " ".join(f"{e:>30}" for e, *_ in SCORE_ERAS))
    for name, fired in _setups(full).items():
        cells = []
        for _, start, end in SCORE_ERAS:
            m = era(full.index, start, end) & fired.fillna(False) & fwd["ret"].notna()
            r = fwd.loc[m, "ret"]
            cells.append(f"{int(m.sum()):>4}d/{signals.count_episodes(m):>3}ep {(r > 0).mean():>4.0%} {r.median():>+6.1%} {fwd.loc[m, 'fell'].mean():>4.0%}"
                         if m.any() else "n/a")
        print(f"  {name:<36} " + " ".join(f"{c:>30}" for c in cells))

    sc = score.combine(st)
    print()
    print(f"What followed each score bucket (the score as weighted), {args.horizon} days on")
    print(f"  {'bucket':<20} {'era':<8} {'days':>5} {'episodes':>9} {'higher':>7} {'median':>8} {'fell 10%':>9}")
    for lo, hi, name in score.BUCKETS:
        for e, start, end in SCORE_ERAS:
            m = era(full.index, start, end) & sc.notna() & fwd["ret"].notna() & (sc > lo) & (sc <= hi)
            if not m.any():
                continue
            r = fwd.loc[m, "ret"]
            print(f"  {name:<20} {e:<8} {int(m.sum()):>5} {signals.count_episodes(m):>9} {(r > 0).mean():>7.0%} "
                  f"{r.median():>+8.1%} {fwd.loc[m, 'fell'].astype(bool).mean():>9.0%}")
    close = full["close"]
    recent = full.index >= pd.Timestamp(score.RECENT, tz="UTC")
    print()
    print("The score against ETH's price")
    print("  Echo (today's score with the return over the past days): "
          + ", ".join(f"{h}d {ic:+.2f}" for h, ic in score.echo(close, sc).items()))
    noise = score.chance_leads(cfg, full)
    leads = {"all": score.lead(close, sc), "to 2020": score.lead(close, sc, mask=pd.Series(~recent, index=full.index)),
             "2021 on": score.lead(close, sc, mask=pd.Series(recent, index=full.index))}
    print("  Lead (today's score with the return over the next days), and chance's spread (one sd, days shuffled):")
    print(f"    {'days ahead':<12}" + "".join(f"{h:>8}" for h in score.LEADS))
    for era, row in leads.items():
        print(f"    {era:<12}" + "".join(f"{row[h]:>+8.2f}" for h in score.LEADS))
    print(f"    {'chance +/-':<12}" + "".join(f"{noise[h]:>8.2f}" for h in score.LEADS))
    print("  Acting on an old score (its age, with the next 30 days from today): "
          + ", ".join(f"{k}d old {ic:+.2f}" for k, ic in score.stale(close, sc).items()))
    print(f"  The score changes sign about {score.flips_per_year(sc):.0f} times a year.")
    rows = score.turns(close, sc)
    if rows:
        print()
        print(f"Turning points ({score.SWING:.0%} swings, found with hindsight): how many days the score took to cross zero "
              "the new way (negative = before the turn), and how much of the swing was still to come")
        print(f"  {'date':<11} {'turn':<7} {'price':>9} {'flipped':<11} {'lag':>5} {'by then':>8} {'swing':>8} {'still to come':>14}")
        for r in rows:
            if r["flip"] is None:
                print(f"  {r['date']:%Y-%m-%d}  {r['kind']:<7} {r['price']:>9,.2f} {'did not turn for this swing':<38}"
                      f"{r['swing']:>+8.1%}" + ("" if r["complete"] else "  (running)"))
                continue
            print(f"  {r['date']:%Y-%m-%d}  {r['kind']:<7} {r['price']:>9,.2f} {r['flip']:%Y-%m-%d}  {r['lag']:>4}d "
                  f"{r['at_flip']:>+8.1%} {r['swing']:>+8.1%} {r['left']:>13.0%}" + ("" if r["complete"] else "  (running)"))
        for kind in ("top", "bottom"):
            done = [r for r in rows if r["kind"] == kind and r["complete"] and r["flip"] is not None]
            sat = sum(1 for r in rows if r["kind"] == kind and r["flip"] is None)
            if done:
                print(f"  {kind}s: median lag {np.median([r['lag'] for r in done]):.0f} days; median share of the swing still "
                      f"to come when the score flipped {np.nanmedian([r['left'] for r in done]):.0%} ({len(done)} turns"
                      + (f"; it did not turn at all for {sat} more" if sat else "") + ")")
    today = sc.iloc[-1]
    print()
    print(f"Today: score {today:+.2f} ({score.label(today)}). Parts: "
          + ", ".join(f"{k} {int(v):+d}" for k, v in st.iloc[-1].items() if pd.notna(v)))
    return 0


DIP_SIZINGS = (("spot, no leverage", {"spot": True}), ("1x perp", {"leverage": 1.0}), ("2.5x perp", {"leverage": 2.5}),
               ("3x perp", {"leverage": 3.0}),
               (f"floor {dipbuy.FLOOR_UNDER:.0%} under start", {"under": dipbuy.FLOOR_UNDER}),
               ("floor, reset on breach", {"under": dipbuy.FLOOR_UNDER, "reset": True}))


def cmd_dip_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(args, ctx, "add cash every week, or save it for a 20% dip, on a "
                                                            "leveraged perp?")
    study = dipbuy.Study(full, swing.daily_funding(full, cfg, funding8h), plan.PLAN_MMR, cfg.taker_fee_rate)
    print(f"Dips: the first close 20% under the 30-day high, {int(study.dips.sum())} in the history. Maintenance margin "
          f"{plan.PLAN_MMR:.0%}, whole-account liquidation at the day's low; saved cash sits in the account as margin.")
    depth = swing.today_depth(full)
    recent = pd.Timestamp(score.RECENT, tz="UTC")
    since = pd.Timestamp(args.since, tz="UTC") if args.since else None
    regular = [d for d in pullbacks.regular_starts(full, years=args.years) if since is None or d >= since]
    deep = [d for d in cycles.deep_starts(full, depth=depth, years=args.years) if since is None or d >= since]

    def eras(starts):
        if since is not None:
            return ((f"{since:%Y} on", starts),)
        return (("all", starts), ("to 2020", [d for d in starts if d < recent]), ("2021 on", [d for d in starts if d >= recent]))

    if since is not None:
        print(f"Only windows starting {since:%Y-%m-%d} or later (indicators still use the earlier history to warm up).")
    groups = (
        ("from a start every 4 weeks", regular, eras(regular), None),
        (f"from every week {asset} closed {depth:.0%}+ below its high (like today)", deep, eras(deep),
         len(cycles.episodes(deep)) if deep else 0),
    )
    for title, starts, eras, drawdowns in groups:
        if not starts:
            continue
        print()
        print(f"{args.years}-year windows {title}: {len(starts)} windows"
              + (f" in {drawdowns} separate drawdowns" if drawdowns else "")
              + ". One unit deposited every week; final equity per unit put in.")
        print(f"  {'sizing':<24} {'starts':<8} {'n':>3} {'weekly':>7} {'dip':>7} {'dip beat':>9} {'above 1x w/d':>13} "
              f"{'liquidated w/d':>15} {'worst w/d':>12} {'dip idle':>9} {'peak lev w/d':>13}")
        for label, kw in DIP_SIZINGS:
            pairs = {d: study.pair(d, args.years, **kw) for d in starts}
            for era, subset in eras:
                if not subset:
                    continue
                sm = dipbuy.summary([pairs[d] for d in subset])
                print(f"  {label:<24} {era:<8} {sm['windows']:>3} {sm['weekly_median']:>6.2f}x {sm['dip_median']:>6.2f}x "
                      f"{sm['dip_beat']:>9.0%} {sm['weekly_above']:>5.0%} / {sm['dip_above']:<5.0%} "
                      f"{sm['weekly_liquidated']:>7.0%} / {sm['dip_liquidated']:<5.0%} "
                      f"{sm['weekly_worst']:>5.2f} / {sm['dip_worst']:<5.2f} {sm['dip_idle']:>9.0%} "
                      f"{sm['weekly_peak_leverage']:>5.1f} / {sm['dip_peak_leverage']:<5.1f}")
    print()
    print("weekly / dip: median final equity per unit put in. dip beat: windows where saving for dips ended ahead. "
          "above 1x: windows that ended with more than was put in. "
          "liquidated: windows with at least one whole-account liquidation. dip idle: cash still waiting at the end. "
          "peak lev: median of each window's highest position / equity.")
    return 0


def cmd_cap_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(args, ctx, "the floor plan with a cap on new cash's leverage, fixed "
                                                            "or set by the signal score")
    study = dipbuy.Study(full, swing.daily_funding(full, cfg, funding8h), plan.PLAN_MMR, cfg.taker_fee_rate)
    sc = score.combine(score.states(cfg, full, funding8h)).fillna(0.0).to_numpy()
    caps = {name: (rule(sc) if rule else None) for name, rule in dipbuy.CAP_RULES.items()}
    since = pd.Timestamp(args.since, tz="UTC") if args.since else None
    depth = swing.today_depth(full)
    groups = (("from a start every 4 weeks", pullbacks.regular_starts(full, years=args.years)),
              (f"from every week {asset} closed {depth:.0%}+ below its high (like today)",
               cycles.deep_starts(full, depth=depth, years=args.years)))
    print(f"Weekly deposits, the floor {dipbuy.FLOOR_UNDER:.0%} under each window's first close and reset after a breach; "
          f"maintenance margin {plan.PLAN_MMR:.0%}. The score is the dashboard's, read each day from that day's close.")
    if since is not None:
        print(f"Only windows starting {since:%Y-%m-%d} or later.")
    for title, starts in groups:
        starts = [d for d in starts if since is None or d >= since]
        if not starts:
            continue
        print()
        print(f"{args.years}-year windows {title}: {len(starts)} windows. Final equity per unit put in")
        print(f"  {'sizing':<30} {'median':>7} {'above 1x':>9} {'beat spot':>10} {'liquidated':>11} {'worst':>6} {'peak lev':>9}")
        spot = [study.pair(d, args.years, spot=True).weekly.multiple for d in starts]
        print(f"  {'spot, no leverage':<30} {np.median(spot):>6.2f}x {np.mean(np.array(spot) > 1):>9.0%} {'':>10} "
              f"{0:>11.0%} {min(spot):>6.2f} {1.0:>9.1f}")
        for name, cap in caps.items():
            runs = [dipbuy.capped(study, d, args.years, dipbuy.FLOOR_UNDER, cap) for d in starts]
            m = np.array([r.multiple for r in runs])
            print(f"  {'floor plan, ' + name:<30} {np.median(m):>6.2f}x {np.mean(m > 1):>9.0%} {np.mean(m > np.array(spot)):>10.0%} "
                  f"{np.mean([r.liquidations > 0 for r in runs]):>11.0%} {m.min():>6.2f} "
                  f"{np.median([r.peak_leverage for r in runs]):>9.1f}")
    print()
    print("beat spot: windows where the plan ended ahead of buying spot every week. liquidated: windows with at least one "
          "breach. peak lev: median of each window's highest position / equity.")
    return 0


ENTRY_FLOORS = (0.40, 0.50)  # floors under the entry close, generic for public logs


def cmd_entry_study(args, ctx: Ctx) -> int:
    asset, full, cfg, _ = _asset_history(args, ctx, "put cash to work now, or bid lower, when each buy is sized "
                                                    "to a liquidation floor?")
    pull = pullbacks.in_pullback(full)
    regimes = (("every 15%+ pullback day", pull), ("pullback days above the 200-day average", pull & cycles.above_trend(full)))
    for under in ENTRY_FLOORS + ((args.under,) if args.under else ()):
        for label, mask in regimes:
            rows = plan.entry_vs_bid(full, under, cfg.maintenance_margin_rate, placed_on=mask)
            if not rows or not rows[0].starts:
                continue
            print()
            print(f"Floor {under:.0%} under the entry-day close; from {label}: {rows[0].starts} starts in {rows[0].episodes} "
                  f"pullbacks. Orders good for {pullbacks.BID_DAYS} days, else bought at the close then. A year on, per dollar")
            print(f"  {'entry':<18} {'filled':>7} {'ETH vs now':>11} {'liquidated':>11}   if not liquidated: {'median value':>12} {'beat now':>9}")
            for r in rows:
                name = "at the close (now)" if r.depth == 0 else f"order {r.depth:.0%} under"
                beat = "" if r.depth == 0 else na(r.beat_now, lambda x: f"{x:.0%}")
                print(f"  {name:<18} {r.filled:>7.0%} {r.eth:>10.2f}x {r.liquidated:>11.0%}   {'':>18} "
                      f"{na(r.value, lambda x: f'{x:.2f}x'):>12} {beat:>9}")
    print()
    print("Sized to the floor, a dollar buys 1 / (price - floor x (1 - maintenance margin)) ETH, so a lower fill buys more ETH "
          "with the same liquidation price. Any fall to the floor fills the order first, so both entries are liquidated on the "
          "same paths. Fees and funding are left out; overlapping starts within one pullback are not independent.")
    return 0


def cmd_price(args, ctx: Ctx) -> int:
    print(f"{_provider(args, ctx).spot():.2f}")
    return 0


def cmd_snapshot(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    df = provider.candles(args.days)
    snap = indicators.snapshot(df, provider.spot())

    def vs(level):
        return "n/a" if level is None else f"{usd(level)}  (price {snap.price / level - 1:+.1%})"

    print(f"ETHUSD {usd(snap.price)}   source: {provider.name}   candles: {snap.candles}")
    print("-" * 78)
    print(f"Returns      7d {na(snap.ret_7d, lambda x: f'{x:+.1%}')}   30d {na(snap.ret_30d, lambda x: f'{x:+.1%}')}   90d {na(snap.ret_90d, lambda x: f'{x:+.1%}')}")
    print(f"SMA 20       {vs(snap.sma_20)}")
    print(f"SMA 50       {vs(snap.sma_50)}")
    print(f"SMA 100      {vs(snap.sma_100)}")
    print(f"SMA 200      {vs(snap.sma_200)}")
    print(f"Volume       {na(snap.volume_avg_20, lambda avg: f'{snap.volume:,.0f} ETH in the latest (possibly partial) day, 20d average {avg:,.0f} ETH')}")
    print(f"RSI 14       {na(snap.rsi_14, lambda x: f'{x:.1f}')}")
    print(f"ATR 14       {na(snap.atr_14, usd)} = {na(snap.atr_14_pct, pct)} of price (typical daily range)")
    print(f"Volatility   30d realised {na(snap.vol_30d, pct)} annualised, typical 1-day move {na(snap.daily_sigma, pct)}")
    print(f"90d range    {usd(snap.low_90d)} - {usd(snap.high_90d)}; {pct(snap.drawdown_from_90d_high)} below the 90d high")
    print(f"Pullback     {pct(snap.drawdown_from_30d_high)} below the 30d high, {pct(snap.rise_from_30d_low)} above the 30d low, {na(snap.zscore_20, lambda x: f'{x:+.1f}')} std devs from the 20d mean")
    print()
    for side in ("long", "short"):
        for line in history_lines(df, side, ctx.cfg.max_holding_days):
            print(line)
    return 0


CANDLE_REFRESH_SECONDS = 300
DERIV_SIGNALS = {"funding", "oi_change", "oi_level", "long_liquidations", "short_liquidations"}
LEVEL_LABELS = [
    ("sma_20", "20d"), ("sma_50", "50d"), ("sma_100", "100d"), ("sma_200", "200d"),
    ("avwap_low", "AVWAP from the low"), ("avwap_year", "AVWAP from the year open"),
    ("val", "value-area low"), ("poc", "point of control"), ("vah", "value-area high"),
]


def _derivs(args, ctx: Ctx) -> pd.DataFrame | None:
    """Derivatives history: an explicit CSV, else what `record` has accumulated, else none."""
    if args.derivs_csv:
        return derivs.load_csv(args.derivs_csv)
    path = derivs.recorded_path(ctx.home / "derivs", ctx.cfg.derivs_instrument)
    return derivs.load_csv(path) if path.exists() else None


def _signals(args, ctx: Ctx, df: pd.DataFrame, price: float) -> signals.SignalSet:
    anchor = pd.Timestamp(args.anchor, tz="UTC") if getattr(args, "anchor", None) else None
    return signals.compute(ctx.cfg, indicators.with_price(df, price), _derivs(args, ctx), anchor)


def print_signals(sigs: signals.SignalSet) -> None:
    def line(g: signals.Signal) -> str:
        mark = {1: "+", -1: "-", 0: "·"}[g.bias] + ("!" if g.risk else " ")
        return f"  {mark} {g.headline}" + ("  (unvalidated)" if g.unvalidated else "")

    levels = [f"{label} {sigs.levels[key]:,.0f}" for key, label in LEVEL_LABELS if key in sigs.levels]
    if levels:
        print("Levels       " + "; ".join(levels))
        print()
    technical = [g for g in sigs.signals if g.name not in DERIV_SIGNALS]
    derivative = [g for g in sigs.signals if g.name in DERIV_SIGNALS]
    for title, group in (("Technical", technical), ("Derivatives", derivative)):
        if group:
            print(f"{title}  (+ bullish, - bearish, · neutral, ! raises the odds of a violent move)")
            print("\n".join(line(g) for g in group))
            print()
    votes = sigs.votes()
    by_family = ", ".join(f"{name} {vote:+d}" for name, vote in votes.items() if vote) or "none voting"
    print(f"Net by family (each votes once, so correlated signals do not stack): {by_family}; total {sum(votes.values()):+d}")
    print(f"Risk flags: {sum(g.risk for g in sigs.signals)}")
    for note in sigs.notes:
        print(f"  note: {note}")


def cmd_signals(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    price = provider.spot()
    sigs = _signals(args, ctx, provider.candles(args.days), price)
    print(f"ETHUSD {usd(price)}   source: {provider.name}")
    print("-" * 78)
    print_signals(sigs)
    return 0


def _assess_all(args, ctx: Ctx, provider: Provider, price: float, df: pd.DataFrame | None = None):
    """[(status, assessment)] for every open position at ``price``, plus the signal set used."""
    now = datetime.now(timezone.utc)
    df = provider.candles(args.days) if df is None else df
    sigs = _signals(args, ctx, df, price)
    statuses = [risk.mark_position(ctx.cfg, p, price, now) for p in ctx.journal.open_positions()]
    book: tuple[str, ...] = ()
    if statuses and ctx.journal.equity is not None:
        book = risk.portfolio_status(ctx.cfg, ctx.journal.equity, ctx.journal.peak_equity, statuses).flags
    priced = indicators.with_price(df, price)
    return [(s, advice.assess(ctx.cfg, s.position, s, sigs, priced, book)) for s in statuses], sigs


def print_assessment(status: risk.PositionStatus, a: advice.Assessment, max_days: int) -> None:
    p = status.position
    print(f"#{p.id} {p.side.upper()} {p.qty:.4f} ETH @ {usd(p.entry)}   mark {usd(status.price)}   {a.move:+.1%} from entry   "
          f"uPnL {signed_usd(status.unrealized)}   held {status.days_held:.1f}/{max_days}d")
    action = a.action + (f"  (close {a.suggested_fraction:.0%})" if a.action == "TAKE_PARTIAL" else "")
    if a.suggested_stop is not None:
        action += f"   suggested stop {usd(a.suggested_stop)} (now {usd(p.stop)})"
    print(f"   ACTION: {action}")
    for reason in a.reasons:
        print(f"     why: {reason}")
    print(f"   Signals: net {a.net_score:+d} by family vote, {a.risk_flags} risk flag(s)")
    print("   Bull case:")
    for item in a.bull_case or ("nothing supportive right now",):
        print(f"     + {item}")
    print("   Bear case:")
    for item in a.bear_case or ("nothing flagged",):
        print(f"     - {item}")


def cmd_advise(args, ctx: Ctx) -> int:
    positions = ctx.journal.open_positions()
    if not positions:
        print("No open positions.")
        return 0
    provider = _provider(args, ctx)
    price = args.price if args.price is not None else provider.spot()
    results, sigs = _assess_all(args, ctx, provider, price)
    print(f"ETHUSD {usd(price)}   source: {provider.name}")
    print("-" * 78)
    for status, a in results:
        print_assessment(status, a, ctx.cfg.max_holding_days)
        print()
    for note in sigs.notes:
        print(f"note: {note}")
    return 1 if any(a.action in ("EXIT", "TAKE_PARTIAL") for _, a in results) else 0


def cmd_record(args, ctx: Ctx) -> int:
    instrument = args.instrument or ctx.cfg.derivs_instrument
    path = derivs.recorded_path(ctx.home / "derivs", instrument)
    try:
        while True:
            row = derivs.fetch_deribit_snapshot(instrument)
            wrote = derivs.append_snapshot(path, row)
            print(f"{row['time']}  {instrument}  OI {row['open_interest']:,.0f}  funding/8h {row['funding_8h']:.4%}"
                  f"{'' if wrote else '  (duplicate timestamp, skipped)'}")
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def _record_derivs(ctx: Ctx) -> None:
    """Append one Deribit open-interest/funding snapshot; a failure is a warning, never fatal."""
    try:
        derivs.append_snapshot(
            derivs.recorded_path(ctx.home / "derivs", ctx.cfg.derivs_instrument),
            derivs.fetch_deribit_snapshot(ctx.cfg.derivs_instrument),
        )
    except DataError as exc:
        print(f"warning: could not record derivatives data ({exc})", file=sys.stderr)


DERIVS_HISTORY_ROWS = 2100  # after compaction: a row a day for years, plus the last 72 hours hourly
JOB_SECONDS = 3600  # the scheduled job runs hourly; the page warns when an update is overdue


CONTEXT_DAYS = 4000  # the trend, the signals' odds and the floor risk are measured over ETH's whole history


def _recent_funding(ctx: Ctx, instrument: str) -> pd.Series | None:
    """About a year of Deribit's daily funding, fetched at most once a day; None (with a warning) if unavailable."""
    path = ctx.home / "funding" / f"deribit-{instrument}-year.csv"
    now = datetime.now(timezone.utc)
    try:
        if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date() == now.date():
            cached = pd.read_csv(path, index_col=0).iloc[:, 0]
            cached.index = pd.to_datetime(cached.index, utc=True)
            return cached.astype(float)
    except (OSError, ValueError, IndexError):
        pass
    try:
        series, _ = derivs.fetch_funding_history(pd.Timestamp(now) - pd.Timedelta(days=366), pd.Timestamp(now), instrument=instrument)
    except DataError as exc:
        print(f"warning: funding history unavailable ({exc})", file=sys.stderr)
        return None
    series = series.tail(365)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("funding_8h").to_csv(path)
    return series


def _exchange_account() -> dict | None:
    """Your Coinbase position and fills, if COINBASE_API_KEY_JSON is set; None (with a warning that names no account
    detail) if it is set but cannot be read."""
    try:
        key = coinbase.env_key()
        return coinbase.account(coinbase.Client(key)) if key else None
    except (DataError, ValueError) as exc:
        print(f"warning: exchange account unavailable ({exc.__class__.__name__}: {coinbase._scrub(str(exc))})", file=sys.stderr)
        return None


def _observe(base: str, token: str, account: dict) -> str | None:
    """Fold this run's exchange positions into the stored record and give each position whose trade history the
    API does not return the trades worked out from it. Returns the record to store, or None if it cannot be read."""
    try:
        state = observed.load(snapshot.get_state(base, token, "trades"))
    except DataError as exc:  # names no account detail: the message could carry the dashboard's address
        print(f"warning: stored exchange positions unavailable ({exc.__class__.__name__})", file=sys.stderr)
        return None
    observed.update(state, account.get("positions") or [], datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for p in account.get("positions") or []:
        if not p.get("trades_source"):
            trades = observed.open_trades(state, p["product"])
            if trades:
                p["trades"], p["trades_source"], p["trades_complete"] = trades, "observed", True
    return observed.dump(state)


def cmd_account_check(args, ctx: Ctx) -> int:
    """Which Coinbase endpoints the key can read, as shapes only: safe for the public daily-job logs."""
    key = coinbase.env_key()
    if key is None:
        raise CliError(f"set {coinbase.ENV} to the contents of your view-only CDP API key file")
    print(f"Coinbase API, {key.alg} key. Shapes only: no amounts, identifiers or counts.")
    for line in coinbase.probe(coinbase.Client(key)):
        print(f"  {line}")
    return 0


def _market_context(ctx: Ctx, provider: Provider) -> dict:
    """Long daily history and a year of funding, for the trend, odds, floor risk and crowding panels, and your
    exchange account when a key is set."""
    # the inverse perpetual's funding: the history endpoint the studies verified, and a percentile only needs its own year
    return {"history": provider.candles(CONTEXT_DAYS), "funding8h": _recent_funding(ctx, derivs.HISTORY_INSTRUMENT),
            "account": _exchange_account()}


def _snapshot(args, ctx: Ctx, provider: Provider, price: float, df: pd.DataFrame, derivs_df=None, **kwargs) -> dict:
    anchor = pd.Timestamp(args.anchor, tz="UTC") if getattr(args, "anchor", None) else None
    derivs_df = _derivs(args, ctx) if derivs_df is None else derivs_df
    return snapshot.build(ctx.cfg, ctx.journal, df, price, derivs_df, anchor=anchor, source=provider.name,
                          basis=getattr(args, "basis", "close"), **kwargs)


def _dashboard_credentials(args) -> tuple[str, str]:
    base = args.url or os.environ.get("ETHTRACKER_DASHBOARD_URL")
    token = os.environ.get("ETHTRACKER_INGEST_TOKEN")
    if not base:
        raise CliError("give --url, or set ETHTRACKER_DASHBOARD_URL to your dashboard's address")
    if not token:
        raise CliError("set ETHTRACKER_INGEST_TOKEN to the dashboard's INGEST_TOKEN (it is read from the environment only)")
    return snapshot.dashboard_base(base), token


def _derivs_from_text(text: str | None) -> pd.DataFrame | None:
    """Derivatives history held by the dashboard, parsed with the same loader as a local file."""
    if not text:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "derivs.csv"
        path.write_text(text)
        try:
            return derivs.load_csv(path)
        except DataError as exc:
            print(f"warning: ignoring unreadable derivatives history ({exc})", file=sys.stderr)
            return None


def cmd_dashboard_json(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    price = args.price if args.price is not None else provider.spot()
    snap = _snapshot(args, ctx, provider, price, provider.candles(args.days), **_market_context(ctx, provider))
    text = json.dumps(snap, indent=2, allow_nan=False)  # strict JSON: a stray NaN is an error, not a broken page
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out} ({len(snap['positions'])} position(s), {len(snap['signals'])} signal(s))")
    else:
        print(text)
    return 0


def cmd_publish(args, ctx: Ctx) -> int:
    """Publish from this machine and push the journal, so the daily job marks the same positions."""
    base, token = _dashboard_credentials(args)
    endpoint = snapshot.ingest_endpoint(base)
    provider = _provider(args, ctx)
    refresh = JOB_SECONDS if args.once else int(args.interval)  # after --once, the scheduled job is the cadence
    df, context, candles_at, first = None, {}, float("-inf"), True
    try:
        while True:
            try:
                price = provider.spot()
                if df is None or time.monotonic() - candles_at > CANDLE_REFRESH_SECONDS:
                    df, candles_at = provider.candles(args.days), time.monotonic()
                    context = _market_context(ctx, provider)
                history = None if args.derivs_csv else _derivs_from_text(snapshot.get_state(base, token, "derivs"))
                snap = _snapshot(args, ctx, provider, price, df, history, refresh_seconds=refresh, publisher="local", **context)
                snapshot.publish(endpoint, token, snap)
                if ctx.journal.path.exists():
                    snapshot.put_state(base, token, "journal", ctx.journal.path.read_text())
                actions = ", ".join(f"#{p['id']} {p['action']}" for p in snap["positions"]) or "no open positions"
                print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  published ETHUSD {usd(price)}  ({actions})")
            except DataError as exc:
                if first:
                    raise
                print(f"warning: not published this round ({exc})", file=sys.stderr)
            first = False
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_cloud_run(args, ctx: Ctx) -> int:
    """The daily job. Pulls your journal and the derivatives history from the dashboard, records today's
    open interest and funding, computes the snapshot on completed daily candles and publishes it.
    Its output is safe for public CI logs: market facts and counts only."""
    base = token = None
    if not args.dry_run:
        base, token = _dashboard_credentials(args)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        journal_text = snapshot.get_state(base, token, "journal") if base else None
        if journal_text is not None:
            (work / "state.json").write_text(journal_text)
        try:
            journal = Journal(work / "state.json")
        except (KeyError, TypeError, ValueError) as exc:
            raise CliError("the journal stored on the dashboard is not valid: run `ethtracker publish --once` to replace it") from exc

        history_path = work / "derivs.csv"
        history = snapshot.get_state(base, token, "derivs") if base else None
        if history:
            history_path.write_text(history)
        recorded = False
        try:
            recorded = derivs.append_snapshot(history_path, derivs.fetch_deribit_snapshot(ctx.cfg.derivs_instrument))
        except DataError as exc:
            print(f"warning: derivatives snapshot unavailable ({exc})", file=sys.stderr)
        derivs_df = _derivs_from_text(history_path.read_text()) if history_path.exists() else None

        provider = _provider(args, ctx)
        price, df = provider.spot(), provider.candles(args.days)
        publisher = "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "cloud-run"
        context = _market_context(ctx, provider)
        trades_doc = _observe(base, token, context["account"]) if base and context.get("account") else None
        snap = snapshot.build(ctx.cfg, journal, df, price, derivs_df, source=provider.name, basis="close",
                              refresh_seconds=JOB_SECONDS, publisher=publisher, **context)
        print(snapshot.public_summary(snap))
        if not args.dry_run:
            print("journal: " + ("pulled from the dashboard" if journal_text is not None
                                 else "none stored yet, so a market-only snapshot (run `ethtracker publish --once` on your machine)"))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(snap, indent=2, allow_nan=False) + "\n")
        if args.dry_run:
            print("dry run: computed from live data, nothing published")
            return 0
        snapshot.publish(snapshot.ingest_endpoint(base), token, snap)
        if recorded:
            snapshot.put_state(base, token, "derivs", snapshot.tail_csv(derivs.compact(history_path.read_text()), DERIVS_HISTORY_ROWS))
        print("published the snapshot" + (" and the derivatives history" if recorded else ""))
        if trades_doc is not None:
            try:
                snapshot.put_state(base, token, "trades", trades_doc)
                print("stored the exchange position for the next run's trade detection")
            except DataError as exc:  # a dashboard deployed before the "trades" state existed refuses it
                print(f"warning: exchange position not stored ({exc.__class__.__name__}); redeploy the dashboard", file=sys.stderr)
    return 0


def _cycle_table(title: str, rows: list, width: int = 40) -> None:
    spot = rows[0].multiple
    print()
    print(title)
    print(f"  {'strategy':<{width}} {'x start':>12} {'vs spot':>8} {'max DD':>7} {'trades':>6} {'in mkt':>6}  liquidated")
    for r in rows:
        vs = f"{r.multiple / spot:.2f}x" if spot > 0 else "n/a"
        liq = (f"account {r.liquidated:%Y-%m-%d}" if r.liquidated is not None
               else f"{r.lost_positions} position(s)" if r.lost_positions else "")
        print(f"  {r.name:<{width}} {r.multiple:>11,.2f}x {vs:>8} {r.max_drawdown:>7.0%} {r.trades:>6} {r.exposure:>6.0%}  {liq}")


def _rolling_table(full: pd.DataFrame, cfg) -> None:
    close = full["close"]
    peak_day = close.idxmax()
    print()
    print(f"Today: {1 - close.iloc[-1] / close.max():.0%} below the highest close ({close.max():,.2f} on {peak_day:%Y-%m-%d}).")
    starts = cycles.deep_starts(full)
    if not starts:
        print(f"No day {cycles.DEEP:.0%}+ below the high has {cycles.HORIZON_YEARS} years of data after it.")
        return
    groups = cycles.episodes(starts)
    spans = ", ".join(f"{g[0]:%Y-%m} to {g[-1]:%Y-%m}" for g in groups)
    print(f"From every week ETH closed {cycles.DEEP:.0%}+ below its highest close so far, {cycles.HORIZON_YEARS} years forward, "
          f"no hindsight: {len(starts)} starts in {len(groups)} separate drawdowns ({spans})")
    print(f"  {'strategy':<40} {'median x':>10} {'beat spot':>9} {'liquidated':>10} {'worst':>7}  median vs spot, each drawdown")
    for r in cycles.rolling(full, cfg, starts):
        each = "  ".join(f"{v:>6.2f}x" for v in r.median_vs_spot_by_episode)
        print(f"  {r.name:<40} {r.median_multiple:>9,.2f}x {r.beat_spot:>9.0%} {r.liquidated:>10.0%} {r.worst:>6.2f}x  {each}")


def cmd_cycle_study(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    raw = provider.candles(args.days)
    full, flashes = cycles.ignore_flash_prints(raw)
    rules = cycles.Rules.of(full)
    cfg = ctx.cfg
    first, last = full.index[0], full.index[-1]
    print("Cycle study: does buying the end of a large dip, with leverage, beat holding spot?")
    print(f"Data: {len(full)} daily candles, {first:%Y-%m-%d} to {last:%Y-%m-%d} ({provider.name}). "
          f"Funding {cfg.funding_rate_8h:.3%}/8h paid by longs, taker fee {cfg.taker_fee_rate:.2%}, "
          f"maintenance margin {cfg.maintenance_margin_rate:.1%}.")
    for day in flashes:
        i = raw.index.get_loc(day)
        print(f"Ignored for liquidations, as a perp's mark price would: the {day:%Y-%m-%d} low of {raw['low'].iloc[i]:,.2f} "
              f"between closes of {raw['close'].iloc[max(i - 1, 0)]:,.2f} and {raw['close'].iloc[i]:,.2f}.")
    phases = cycles.find_phases(full)
    print()
    print("Cycle phases, with hindsight: each peak is the highest close in its cycle's window, each trough the lowest close before the next")
    print(f"  {'phase':<5} {'from':<10}  {'to':<10} {'from close':>11} {'to close':>10} {'change':>9} {'worst fall inside':>18}")
    for ph in phases:
        note = "  (starts where the data does)" if ph.kind == "bull" and not ph.from_bottom else ""
        note += "  (still running)" if not ph.complete else ""
        print(f"  {ph.kind:<5} {ph.start:%Y-%m-%d}  {ph.end:%Y-%m-%d} {ph.start_close:>11,.2f} {ph.end_close:>10,.2f} "
              f"{ph.change:>+9.0%} {-ph.worst_fall:>18.0%}{note}")

    bulls = []
    for ph in phases:
        if ph.kind != "bull":
            continue
        rows = cycles.run_window(full, ph.start, ph.end, cfg, rules=rules)
        where = "bottom to top" if ph.from_bottom else "start of data to top"
        _cycle_table(f"Bull phase {ph.start:%Y-%m-%d} to {ph.end:%Y-%m-%d} ({where}{', still running' if not ph.complete else ''})", rows)
        if ph.from_bottom and ph.complete:
            bulls.append(rows)
    if len(bulls) > 1:
        _cycle_table(f"The {len(bulls)} bottom-to-top bull phases back to back (out at each top, back in at each bottom)", cycles.chain(bulls))
    current = phases[-1]
    if current.kind == "bear" and not current.complete:
        _cycle_table(f"Current bear phase {current.start:%Y-%m-%d} to {current.end:%Y-%m-%d} (each rule started at the top)",
                     cycles.run_window(full, current.start, current.end, cfg, rules=rules))
    _rolling_table(full, cfg)
    _cycle_table(f"Whole history {first:%Y-%m-%d} to {last:%Y-%m-%d}, no hindsight",
                 cycles.run_window(full, first, last, cfg, rules=rules))
    _cycle_table(f"Whole history, dip entries only when price is above its {cycles.TREND}-day average",
                 cycles.run_window(full, first, last, cfg, gated=True, rules=rules))
    print()
    print(f"Rules: a large dip is a close {cycles.DIP:.0%} below the 30-day high. Dip rules sell when RSI(14) reaches {cycles.EXIT_RSI:.0f} "
          f"or after {cycles.MAX_HOLD} days. Leverage is total exposure / equity in one cross-margin account.")
    print("Hindsight flatters: every bull phase above starts at the exact bottom, which suits holding with leverage most of all.")
    print("Funding is held constant; in bull markets it has run higher, which costs every leveraged row in proportion to time held.")
    print("Trades at the daily close, liquidation at the day's low, no slippage. The spot-core rule assumes spot counts as collateral.")
    print("Overlapping start dates are not independent: the rolling table holds as many real observations as it has drawdowns.")
    print(f"{len(cycles.STRATEGIES)} fixed rules, not tuned. Trust what holds in every phase, not the best row in one.")
    return 0


def _combo_table(title: str, rows: list) -> None:
    spot = rows[0].multiple
    print()
    print(title)
    print(f"  {'strategy':<49} {'x start':>12} {'vs spot':>8} {'max DD':>7} {'core liquidated':>15} "
          f"{'swings':>6} {'swing liq':>9} {'swept':>7}")
    for r in rows:
        vs = f"{r.multiple / spot:.2f}x" if spot > 0 else "n/a"
        dead = f"{r.core_liquidated:%Y-%m-%d}" if r.core_liquidated is not None else ""
        print(f"  {r.name:<49} {r.multiple:>11,.2f}x {vs:>8} {r.max_drawdown:>7.0%} {dead:>15} "
              f"{r.swing_trades:>6} {r.swing_liquidated:>9} {r.swept:>6.2f}x")


def cmd_combo_study(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    raw = provider.candles(args.days)
    full, flashes = cycles.ignore_flash_prints(raw)
    rules = cycles.Rules.of(full)
    cfg = ctx.cfg
    first, last = full.index[0], full.index[-1]
    cost = cfg.funding_rate_8h * risk.FUNDING_PERIODS_PER_DAY * 365
    print("Two sleeves: a long-only perp core, plus a small high-leverage swing sleeve whose profits are swept into the core")
    print(f"Data: {len(full)} daily candles, {first:%Y-%m-%d} to {last:%Y-%m-%d} ({provider.name}). "
          f"Funding {cfg.funding_rate_8h:.3%}/8h ({cost:.1%} a year on the notional), taker fee {cfg.taker_fee_rate:.2%}, "
          f"maintenance margin {cfg.maintenance_margin_rate:.1%}.")
    for day in flashes:
        print(f"Ignored for liquidations, as a perp's mark price would: the {day:%Y-%m-%d} low of {raw.loc[day, 'low']:,.2f}.")

    phases = cycles.find_phases(full)
    bulls = [ph for ph in phases if ph.kind == "bull" and ph.from_bottom and ph.complete]
    periods = [("Whole history", first, last)]
    if bulls:
        periods.append((f"Since the {bulls[0].start:%Y-%m-%d} bottom", bulls[0].start, last))
    periods += [(f"Bull {ph.start:%Y}-{ph.end:%Y}, bottom to top", ph.start, ph.end) for ph in bulls]
    periods.append(("Last 4 years", last - pd.DateOffset(years=4), last))
    print()
    print("The arithmetic of leverage: a position kept at leverage L grows about L(mu - funding) - L^2 sigma^2 / 2 a year")
    print(f"  {'period':<30} {'volatility':>10} {'spot':>8} {'1.4x':>8} {'2.5x':>8} {'growth-optimal':>15}")
    for label, start, end in periods:
        g = combo.growth(full[start:end], label)
        yearly = lambda x: f"{np.expm1(x):+.0%}"
        print(f"  {label:<30} {g.vol:>10.0%} {yearly(g.drift):>8} {yearly(g.at(1.4, cost)):>8} {yearly(g.at(2.5, cost)):>8} "
              f"{g.kelly(cost):>14.2f}x")
    print("  (compound growth a year; growth-optimal is the leverage that maximises it, and growth turns negative at twice that)")

    windows = []
    for ph in bulls:
        rows = combo.run_window(full, ph.start, ph.end, cfg, rules)
        _combo_table(f"Bull phase {ph.start:%Y-%m-%d} to {ph.end:%Y-%m-%d}, bottom to top (hindsight)", rows)
        windows.append(rows)
    if len(windows) > 1:
        _combo_table(f"The {len(windows)} bull phases back to back", combo.chain(windows))
    current = phases[-1]
    if current.kind == "bear" and not current.complete:
        _combo_table(f"Current bear phase {current.start:%Y-%m-%d} to {current.end:%Y-%m-%d}, started at the top",
                     combo.run_window(full, current.start, current.end, cfg, rules))

    starts = cycles.deep_starts(full)
    if starts:
        groups = cycles.episodes(starts)
        print()
        print(f"From every week ETH closed {cycles.DEEP:.0%}+ below its highest close so far, {cycles.HORIZON_YEARS} years forward, "
              f"no hindsight: {len(starts)} starts in {len(groups)} separate drawdowns. Today: "
              f"{1 - full['close'].iloc[-1] / full['close'].max():.0%} below the highest close.")
        print(f"  {'strategy':<49} {'median x':>10} {'beat spot':>9} {'core liq':>8} {'worst':>7}  median vs spot, each drawdown")
        runner = lambda f, s, e, c, r: combo.run_window(f, s, e, c, r)
        for r in cycles.rolling(full, cfg, starts, runner=runner):
            each = "  ".join(f"{v:>5.2f}x" for v in r.median_vs_spot_by_episode)
            print(f"  {r.name:<49} {r.median_multiple:>9,.2f}x {r.beat_spot:>9.0%} {r.liquidated:>8.0%} {r.worst:>6.2f}x  {each}")

        picks = [c for c in combo.COMBOS if c.name in (
            "Hold spot, all in", "85% perp 2.5x held + 15% swing 10x", "85% perp 1.4x held + 15% swing 10x", "85% spot + 15% swing 10x")]
        rates = (0.0, cfg.funding_rate_8h, 3 * cfg.funding_rate_8h)
        table = {}
        for rate in rates:
            cfg_r = replace(cfg, funding_rate_8h=rate)
            runner = lambda f, s, e, c, r: combo.run_window(f, s, e, c, r, combos=picks)
            table[rate] = cycles.rolling(full, cfg_r, starts, runner=runner)
        print()
        print("Funding sensitivity, same windows: median x start (share of windows beating spot)")
        print(f"  {'strategy':<49} " + " ".join(f"{f'{rate:.3%}/8h':>16}" for rate in rates))
        for j, pick in enumerate(picks):
            cells = " ".join(f"{f'{table[rate][j].median_multiple:,.2f}x ({table[rate][j].beat_spot:.0%})':>16}" for rate in rates)
            print(f"  {pick.name:<49} {cells}")

    _combo_table(f"Whole history {first:%Y-%m-%d} to {last:%Y-%m-%d}, no hindsight", combo.run_window(full, first, last, cfg, rules))
    print()
    print("Sleeves are separate margin pools. Each swing trade posts the whole sleeve; on a winning exit the sleeve's excess over")
    print("its starting stake is swept into the core (bought at the core's leverage, or as margin only where stated). A wiped-out")
    print("sleeve is not refilled. 'Held': one perp opened at the start, so leverage falls as price rises and climbs as it falls.")
    print(f"'Kept': resized at the close whenever leverage drifts {combo.BAND:.0%} from target. Swing rule: the dip-done entry,")
    print("out at RSI 70 or 30 days. The growth arithmetic assumes the period's drift continues, which is the real unknown.")
    return 0


FUNDING_HISTORY_FROM = "2019-01-01"


def _funding_history(ctx: Ctx, instrument: str, end: pd.Timestamp) -> tuple[pd.Series, int]:
    """Deribit's daily funding history, fetched at most once a day per instrument (it takes a minute or two)."""
    path = ctx.home / "funding" / f"deribit-{instrument}.csv"
    today = datetime.now(timezone.utc).date()
    try:
        if datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date() == today:
            cached = pd.read_csv(path, index_col=0).iloc[:, 0]
            cached.index = pd.to_datetime(cached.index, utc=True)
            return cached.astype(float), 0
    except (OSError, ValueError, IndexError):
        pass
    series, failed = derivs.fetch_funding_history(pd.Timestamp(FUNDING_HISTORY_FROM, tz="UTC"), end, instrument=instrument)
    if not failed:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename("funding_8h").to_csv(path)
    return series, failed


def _overlay_table(title: str, rows: list) -> None:
    core = rows[0].multiple
    print()
    print(title)
    print(f"  {'strategy':<52} {'x start':>12} {'vs core':>8} {'max DD':>7} {'cuts':>5}  liquidated")
    for r in rows:
        vs = f"{r.multiple / core:.2f}x" if core > 0 else "n/a"
        dead = f"{r.liquidated:%Y-%m-%d}" if r.liquidated is not None else ""
        print(f"  {r.name:<52} {r.multiple:>11,.2f}x {vs:>8} {r.max_drawdown:>7.0%} {r.trades:>5}  {dead}")


def cmd_short_study(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    raw = provider.candles(args.days)
    full, _ = cycles.ignore_flash_prints(raw)
    cfg = ctx.cfg
    first, last = full.index[0], full.index[-1]
    print("Shorting an overheated market: in at the signal's close, covered at the first close below the "
          f"{shorts.COOL_AVG}-day average or after {shorts.HORIZON} days")
    print(f"Candles: {len(full)} days, {first:%Y-%m-%d} to {last:%Y-%m-%d} ({provider.name}). "
          f"Taker fee {cfg.taker_fee_rate:.2%}, maintenance margin {cfg.maintenance_margin_rate:.1%}.")
    funding8h = None
    try:
        funding8h, failed = _funding_history(ctx, derivs.HISTORY_INSTRUMENT, last + pd.Timedelta(days=1))
        funding8h = funding8h[(funding8h.index >= first) & (funding8h.index <= last)]
        print(f"Funding: Deribit {derivs.HISTORY_INSTRUMENT}, {len(funding8h)} days from {funding8h.index[0]:%Y-%m-%d} to "
              f"{funding8h.index[-1]:%Y-%m-%d}; median {funding8h.median():.4%}/8h, mean {funding8h.mean():.4%}/8h, "
              f"{(funding8h >= shorts.FUNDING_HOT).mean():.0%} of days at {shorts.FUNDING_HOT:.2%}+, "
              f"{(funding8h < 0).mean():.0%} negative, highest {funding8h.max():.3%}"
              + (f"; {failed} monthly page(s) failed" if failed else "") + ". "
              f"Elsewhere {cfg.funding_rate_8h:.3%}/8h is assumed.")
    except DataError as exc:
        print(f"Funding history unavailable ({exc}); the funding signal is skipped and {cfg.funding_rate_8h:.3%}/8h is assumed.")

    entries = shorts.overheated(full, cfg, funding8h)
    cool = shorts.cooled(full)
    funding = shorts.daily_funding(full, cfg, funding8h)

    print()
    print(f"What followed each signal over the next {shorts.HORIZON} days (first day of each episode)")
    print(f"  {'signal':<40} {'episodes':>8} {'median ETH move':>15} {'fell 10%+':>9} {'squeezed a 10x short':>20} {'a 15x short':>11}")
    everyday = np.ones(len(full), dtype=bool)
    rows = [shorts.events("Every day (baseline)", full, cfg, everyday)]
    if funding8h is not None:
        since = (full.index >= funding8h.index[0])
        rows.append(shorts.events(f"Every day since {funding8h.index[0]:%Y-%m-%d} (baseline)", full, cfg, everyday & since))
    rows += [shorts.events(name, full, cfg, e) for name, e in entries.items()]
    for ev in rows:
        print(f"  {ev.name:<40} {ev.count:>8} {ev.median_move:>+15.1%} {ev.fell_10:>9.0%} {ev.squeeze[10.0]:>20.0%} {ev.squeeze[15.0]:>11.0%}")

    print()
    print("Short trades on each signal: return on margin per trade, and the whole sleeve re-staked every trade")
    print(f"  {'signal':<40} {'lev':>5} {'trades':>6} {'won':>5} {'liquidated':>10} {'mean':>8} {'median':>8} {'re-staked':>11}")
    for name, e in entries.items():
        for lev in shorts.LEVERAGES:
            r = shorts.ShortResult.of(name, lev, shorts.short_returns(full, cfg, e, cool, funding, lev))
            print(f"  {name:<40} {lev:>4g}x {r.trades:>6} {r.win_rate:>5.0%} {r.liquidated:>10.0%} {r.mean_return:>+8.1%} "
                  f"{r.median_return:>+8.1%} {r.restaked:>10.2f}x")

    overlay = shorts.Overlay(full, cfg, funding8h)
    phases = cycles.find_phases(full)
    for ph in phases:
        if ph.kind == "bull" and ph.from_bottom and ph.complete and ph.end >= pd.Timestamp(FUNDING_HISTORY_FROM, tz="UTC"):
            _overlay_table(f"Your {shorts.CORE_LEVERAGE:g}x core, bull phase {ph.start:%Y-%m-%d} to {ph.end:%Y-%m-%d} (bottom to top, hindsight)",
                           overlay.run(full, ph.start, ph.end, cfg))
    current = phases[-1]
    if current.kind == "bear" and not current.complete:
        _overlay_table(f"Your {shorts.CORE_LEVERAGE:g}x core, current bear phase {current.start:%Y-%m-%d} to {current.end:%Y-%m-%d} (from the top)",
                       overlay.run(full, current.start, current.end, cfg))
    era = pd.Timestamp(FUNDING_HISTORY_FROM, tz="UTC")
    _overlay_table(f"Your {shorts.CORE_LEVERAGE:g}x core, {era:%Y-%m-%d} to {last:%Y-%m-%d} (no hindsight)", overlay.run(full, era, last, cfg))
    starts = [d for d in cycles.deep_starts(full) if d >= era]
    if starts:
        groups = cycles.episodes(starts)
        print()
        print(f"Your {shorts.CORE_LEVERAGE:g}x core from every week ETH closed {cycles.DEEP:.0%}+ below its high since {era:%Y}, "
              f"{cycles.HORIZON_YEARS} years forward: {len(starts)} starts in {len(groups)} separate drawdowns")
        print(f"  {'strategy':<52} {'median x':>10} {'beat core':>9} {'liquidated':>10} {'worst':>7}")
        for r in cycles.rolling(full, cfg, starts, runner=overlay.run):
            print(f"  {r.name:<52} {r.median_multiple:>9,.2f}x {r.beat_spot:>9.0%} {r.liquidated:>10.0%} {r.worst:>6.2f}x")
    print()
    print("A short opens at the signal day's close; a later signal while it is open is skipped. Liquidation at the day's high,")
    print("no slippage. Funding is Deribit's inverse ETH perpetual, a proxy for your venue. In one account a short on the same")
    print(f"perp nets against the long, so the overlay cuts the core to {shorts.DERISK_TO:g}x and buys the same quantity back when it cools.")
    return 0


def _asset_history(args, ctx: Ctx, title: str):
    """Candles and Deribit funding for --asset, with the header lines every per-asset study prints."""
    asset = args.asset.upper()
    product, instrument = swing.INSTRUMENTS[asset]
    provider = _provider(args, ctx, product)
    raw = provider.candles(args.days)
    full, flashes = cycles.ignore_flash_prints(raw)
    cfg = ctx.cfg
    print(f"{asset}: {title}")
    print(f"Candles: {len(full)} days of {product}, {full.index[0]:%Y-%m-%d} to {full.index[-1]:%Y-%m-%d} ({provider.name}). "
          f"Taker fee {cfg.taker_fee_rate:.2%}, maintenance margin {cfg.maintenance_margin_rate:.1%}.")
    for day in flashes:
        print(f"Ignored for liquidations, as a perp's mark price would: the {day:%Y-%m-%d} low of {raw.loc[day, 'low']:,.2f}.")
    funding8h = None
    try:
        funding8h, failed = _funding_history(ctx, instrument, full.index[-1] + pd.Timedelta(days=1))
        funding8h = funding8h[(funding8h.index >= full.index[0]) & (funding8h.index <= full.index[-1])]
        print(f"Funding: Deribit {instrument}, {len(funding8h)} days from {funding8h.index[0]:%Y-%m-%d}; median {funding8h.median():.4%}/8h, "
              f"mean {funding8h.mean():.4%}/8h, {(funding8h < 0).mean():.0%} of days negative"
              + (f"; {failed} monthly page(s) failed" if failed else "") + f". Before that {cfg.funding_rate_8h:.3%}/8h is assumed.")
    except DataError as exc:
        print(f"Funding history unavailable ({exc}); {cfg.funding_rate_8h:.3%}/8h is assumed throughout.")
    now = swing.today(full)
    print(f"Today: {now.close:,.2f}, {now.below_high:.0%} below the highest close, {now.below_30d_high:.0%} below the 30-day high, "
          f"RSI {now.rsi:.0f}, {now.run_14d:+.0%} on the 14-day low close.")
    return asset, full, cfg, funding8h


def _asset_tables(asset: str, full: pd.DataFrame, cfg, run, width: int = 40) -> None:
    """The same four views for any per-asset study: whole history, bull phases, the current bear, 4-year windows."""
    first, last = full.index[0], full.index[-1]
    phases = cycles.find_phases(full)
    bulls = [ph for ph in phases if ph.kind == "bull" and ph.from_bottom and ph.complete]
    _cycle_table(f"Whole history {first:%Y-%m-%d} to {last:%Y-%m-%d}, no hindsight", run(full, first, last, cfg), width)
    windows = [run(full, ph.start, ph.end, cfg) for ph in bulls]
    if windows:
        spans = ", ".join(f"{ph.start:%Y-%m-%d} to {ph.end:%Y-%m-%d}" for ph in bulls)
        _cycle_table(f"Bull phases only, bottom to top, back to back (hindsight): {spans}",
                     cycles.chain(windows) if len(windows) > 1 else windows[0], width)
    current = phases[-1]
    if current.kind == "bear" and not current.complete:
        _cycle_table(f"Current bear phase {current.start:%Y-%m-%d} to {current.end:%Y-%m-%d} (from the top)",
                     run(full, current.start, current.end, cfg), width)
    depth = swing.today_depth(full)
    starts = cycles.deep_starts(full, depth=depth)
    if starts:
        groups = cycles.episodes(starts)
        print()
        print(f"From every week {asset} closed {depth:.0%}+ below its highest close so far (today: {swing.today(full).below_high:.0%}), "
              f"{cycles.HORIZON_YEARS} years forward, no hindsight: {len(starts)} starts in {len(groups)} separate drawdowns")
        print(f"  {'strategy':<{width}} {'median x':>10} {'beat spot':>9} {'liquidated':>10} {'worst':>7}")
        for r in cycles.rolling(full, cfg, starts, runner=run):
            print(f"  {r.name:<{width}} {r.median_multiple:>9,.2f}x {r.beat_spot:>9.0%} {r.liquidated:>10.0%} {r.worst:>6.2f}x")


def cmd_swing_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(
        args, ctx, "buy a sharp pullback, sell only into a spike. One position at a time, no stop besides liquidation.")
    rules = swing.Rules(full, cfg, funding8h)
    first, last = full.index[0], full.index[-1]
    phases = cycles.find_phases(full)
    bulls = [ph for ph in phases if ph.kind == "bull" and ph.from_bottom and ph.complete]
    periods = [("Whole history", first, last)]
    if bulls:
        periods.append((f"Since the {bulls[0].start:%Y-%m-%d} bottom", bulls[0].start, last))
    periods += [(f"Bull {ph.start:%Y}-{ph.end:%Y}, bottom to top", ph.start, ph.end) for ph in bulls]
    periods.append(("Last 4 years", last - pd.DateOffset(years=4), last))
    print()
    print("Volatility and the leverage that maximises growth (funding at each period's average)")
    print(f"  {'period':<30} {'volatility':>10} {'spot':>8} {'2.5x':>8} {'growth-optimal':>15}")
    for label, start, end in periods:
        g = combo.growth(full[start:end], label)
        cost = rules.cost(start, end)
        print(f"  {label:<30} {g.vol:>10.0%} {np.expm1(g.drift):>+8.0%} {np.expm1(g.at(2.5, cost)):>+8.0%} {g.kelly(cost):>14.2f}x")
    print("  (spot and 2.5x as compound growth a year)")
    _asset_tables(asset, full, cfg, rules.run)
    print()
    print(f"Entries: the first close {{20%, 25%}} below the {cycles.DIP_LOOKBACK}-day high, or the first close back above the "
          f"10-day average within {cycles.CONFIRM_WITHIN} days of a 20% dip.")
    print(f"Exits: RSI(14) at {swing.SPIKE_RSI:.0f}+, or a close {swing.RUN:.0%}+ above the lowest close of the last {swing.RUN_DAYS} days. "
          "No time stop, no price stop: liquidation is the only other way out.")
    return 0


def cmd_trim_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(
        args, ctx, "trim into spikes and refill on dips, with the liquidation price pinned where it started.")
    rules = swing.PinnedRules(full, cfg, funding8h)
    now = swing.today(full)
    mmr = cfg.maintenance_margin_rate
    print("Pinned liquidation prices if opened today: " + ", ".join(
        f"{lev:g}x at {swing.pinned_floor(now.close, lev, mmr):,.0f}" for lev in (2.5, 2.0, 1.5)) + ".")
    _asset_tables(asset, full, cfg, rules.run, width=58)
    print()
    print("Pinned: the position opens at the stated leverage and its liquidation price is never allowed above that first level;")
    print("funding creep is trimmed away each close. Trims sell a share on the first day of a spike (RSI 80+, or a close 30%+ above")
    print("the 14-day low close) and keep the cash as margin; refills buy back up to the pinned liquidation price on the first")
    print("close 20% below the 30-day high. You end up holding more only when a refill happens below the price of the trim before it.")
    return 0


def _calibration_table(title: str, rows: list) -> None:
    print()
    print(title)
    print(f"  {'signal':<42} {'events':>6} {'low held 30d':>12} {'(95% low)':>9} {'higher 30d':>10} {'90d':>5} "
          f"{'365d':>5} {'(n)':>5} {'median 90d':>10}")
    for c in rows:
        print(f"  {c.name:<42} {c.n:>6} {c.held:>12.0%} {c.held_low:>9.0%} {c.higher[30]:>10.0%} {c.higher[90]:>5.0%} "
              f"{c.higher[365]:>5.0%} {c.n_year:>5} {c.median_90:>+10.0%}")


def cmd_pullback_study(args, ctx: Ctx) -> int:
    asset, full, cfg, funding8h = _asset_history(
        args, ctx, "has the pullback run its course? How often 'the low is in' signals were right, and adding every week.")
    found = pullbacks.signals(full, funding8h)
    fills = pullbacks.bid_fills(full)
    up = cycles.above_trend(full)
    pull = pullbacks.in_pullback(full)
    close = full["close"]
    above = pullbacks.above_trend_today(full)
    print(f"Today: {'above' if above else 'below'} the {cycles.TREND}-day average "
          f"({indicators.sma(close, cycles.TREND).iloc[-1]:,.2f}); "
          + ("in a 15%+ pullback" if pull[-1] else "not in a 15%+ pullback")
          + f"; an order {pullbacks.BID:.0%} under today's close would sit at {close.iloc[-1] * (1 - pullbacks.BID):,.2f}.")
    last_fired = []
    for name, mask in found.items():
        hits = np.flatnonzero(mask)
        last_fired.append(f"{name}: {full.index[hits[-1]]:%Y-%m-%d}" if len(hits) else f"{name}: never")
    print("Last fired: " + "; ".join(last_fired) + ".")

    def table(title, keep):
        rows = [pullbacks.calibrate("Every day (baseline)", full, pullbacks.at_close(full, keep)),
                pullbacks.calibrate("Every day in a 15%+ pullback (baseline)", full, pullbacks.at_close(full, keep & pull))]
        rows += [pullbacks.calibrate(name, full, pullbacks.at_close(full, mask & keep)) for name, mask in found.items()]
        rows.append(pullbacks.calibrate(f"Order {pullbacks.BID:.0%} under, filled", full,
                                        [(j, px) for j, px in fills if keep[j]], from_fill=True))
        _calibration_table(title, rows)

    everything = np.ones(len(full), dtype=bool)
    table(f"How often the low held (no further {pullbacks.FURTHER:.0%} fall in {pullbacks.HELD_DAYS} days) and price was higher, "
          "all history", everything)
    regime = up if above else ~up
    table(f"The same, only when price was {'above' if above else 'below'} its {cycles.TREND}-day average, as it is today", regime)

    print()
    print(f"A buy order under the close on every 15%+ pullback day, good for {pullbacks.BID_DAYS} days")
    print(f"  {'order':<14} {'days':>6} {'pullbacks':>9} {'filled':>7} {'low held after':>14} {'higher 90d':>10} {'higher 365d':>11}")
    for r in pullbacks.ladder(full):
        level = close.iloc[-1] * (1 - r.depth)
        print(f"  {f'{r.depth:.0%} under':<14} {r.days:>6} {r.episodes:>9} {r.filled:>7.0%} {r.held:>14.0%} "
              f"{r.higher_90:>10.0%} {r.higher_365:>11.0%}   (today: {level:,.0f})")

    plans = pullbacks.Plans(full, cfg, funding8h)
    for title, starts in (
        (f"Adding one unit every week for {cycles.HORIZON_YEARS} years, from a start every 4 weeks", pullbacks.regular_starts(full)),
        (f"The same, from every week {asset} closed {swing.today_depth(full):.0%}+ below its high (like today)",
         cycles.deep_starts(full, depth=swing.today_depth(full))),
    ):
        if not starts:
            continue
        print()
        print(f"{title}: {len(starts)} starts. Final value per unit put in; saved cash counts at face value")
        print(f"  {'plan':<42} {'median':>8} {'beat DCA':>9} {'worst':>7} {'idle cash left':>15}")
        results = cycles.rolling(full, cfg, starts, runner=plans.run)
        idle = {}
        for day in starts:
            for r in plans.run(full, day, day + pd.DateOffset(years=cycles.HORIZON_YEARS), cfg):
                idle.setdefault(r.name, []).append(r.idle)
        for r in results:
            print(f"  {r.name:<42} {r.median_multiple:>7.2f}x {r.beat_spot:>9.0%} {r.worst:>6.2f}x {np.median(idle[r.name]):>15.0%}")
    print()
    print(f"Signals enter at the close of the day they fire; the order fills at its price when the day's low reaches it.")
    print(f"'Low held': no fall of {pullbacks.FURTHER:.0%} or more below the entry within {pullbacks.HELD_DAYS} days. The 95% low "
          "column is what the sample size can honestly claim.")
    return 0


def cmd_timing_study(args, ctx: Ctx) -> int:
    asset, full, cfg, _ = _asset_history(args, ctx, "buying every week: does picking the day within a few days help, "
                                                    "and how often did price break a floor under the start?")
    timing = pullbacks.Timing(full, cfg)
    for title, starts in (
        (f"Buying one unit every week for {cycles.HORIZON_YEARS} years, from a start every 4 weeks", pullbacks.regular_starts(full)),
        (f"The same, from every week {asset} closed {swing.today_depth(full):.0%}+ below its high (like today)",
         cycles.deep_starts(full, depth=swing.today_depth(full))),
    ):
        if not starts:
            continue
        ratios: dict[str, list[float]] = {}
        for day in starts:
            rows = timing.run(full, day, day + pd.DateOffset(years=cycles.HORIZON_YEARS), cfg)
            for r in rows:
                ratios.setdefault(r.name, []).append(r.multiple / rows[0].multiple - 1)
        print()
        print(f"{title}: {len(starts)} starts. The buy may fall on any of {pullbacks.WINDOW} days each week; "
              "coins bought against buying on the fixed day")
        print(f"  {'rule':<52} {'median':>8} {'better in':>9} {'worst':>7} {'best':>7}")
        for name, values in ratios.items():
            v = np.array(values)
            print(f"  {name:<52} {np.median(v):>+8.1%} {np.mean(v > 0):>9.0%} {v.min():>+7.1%} {v.max():>+7.1%}")

    depth = swing.today_depth(full)
    starts = cycles.deep_starts(full, depth=depth)
    floors = list(pullbacks.FLOORS)
    now = full["close"].iloc[-1]
    if args.floor:
        floors.append(1 - args.floor / now)
    breaches = pullbacks.floor_breaches(full, starts, depths=floors)
    print()
    print(f"How often {asset} traded through a floor under the starting price, from every week it closed {depth:.0%}+ "
          f"below its high ({len(starts)} starts in {len(cycles.episodes(starts))} drawdowns)")
    print(f"  {'floor':<34} " + " ".join(f"{f'within {y}y':>10}" for y in pullbacks.FLOOR_YEARS))
    for d in floors:
        label = f"{d:.0%} under the start" + (f" (yours: {args.floor:,.0f} from {now:,.0f})" if args.floor and d == floors[-1] else "")
        print(f"  {label:<34} " + " ".join(f"{breaches[d][y]:>10.0%}" for y in pullbacks.FLOOR_YEARS))
    if args.floor:
        mmr = cfg.maintenance_margin_rate
        print()
        print(f"Adding new cash with the liquidation price kept at {args.floor:,.0f}: the most leverage each add can carry")
        for move in (-0.2, -0.1, 0.0, 0.3, 0.75):
            price = now * (1 + move)
            print(f"  at {price:>10,.0f} ({move:+.0%} from today): {pullbacks.add_leverage_cap(price, args.floor, mmr):.2f}x")
    print()
    print("Rules decide each day from that day's close only; the hindsight rows show the most and least any rule could do.")
    print("A breached floor liquidates everything added against it, not just the latest add.")
    return 0


def cmd_signals_study(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    df = provider.candles(args.days)
    horizon = args.horizon or ctx.cfg.max_holding_days
    buckets = signals.warning_study(ctx.cfg, df, horizon)
    print(f"Technical risk flags (stretched, RSI hot, volatility spike) versus what followed a long entered at that day's close")
    print(f"Data: {len(df)} daily candles, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({provider.name}); horizon {horizon}d")
    print("-" * 78)
    print(f"  {'bucket':<10} {'days':>5} {'episodes':>9} {'median return':>14} {'>=15% drawdown':>15} {'>=20% drawdown':>15} {'>=+10% rally':>13}")
    for b in buckets:
        episodes = "" if b.label == "all days" else b.episodes
        print(f"  {b.label:<10} {b.n:>5} {episodes:>9} {b.median_return:>+14.1%} {b.drawdown_15:>15.0%} {b.drawdown_20:>15.0%} {b.rally_10:>13.0%}")
    base = next((b for b in buckets if b.label == "all days"), None)
    flagged = next((b for b in buckets if b.label == "2+ flags"), None)
    print()
    if base and flagged and base.drawdown_20 > 0:
        print(f"2+ flags: a >=20% drawdown followed {flagged.drawdown_20:.0%} of the time versus {base.drawdown_20:.0%} for all days "
              f"({flagged.drawdown_20 / base.drawdown_20:.1f}x), on {flagged.n} days in {flagged.episodes} separate episode(s). "
              f"The rally column is the upside an early exit would have given up.")
    elif base:
        print("No qualifying 2+ flag days in this history.")
    print("Windows overlap and flagged days cluster, so n overstates the independent evidence. Judge a lift by n, not just its size.")
    print("Open interest, funding and liquidations are not tested here: free history of them is too short or absent.")
    return 0


def cmd_sync(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    if not isinstance(provider, CachedProvider):
        raise CliError("sync fills the local cache from a network provider: use coinbase or kraken, without --no-cache")
    df = provider.candles(args.days)
    print(f"{provider.name}: {len(df)} daily candles, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}")
    print(f"cache: {provider.path}")
    return 0


def cmd_export(args, ctx: Ctx) -> int:
    out = indicators.frame(_provider(args, ctx).candles(args.days))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index_label="date")
    print(f"wrote {len(out)} rows to {args.out}: date, {', '.join(out.columns)}")
    return 0


def cmd_headroom(args, ctx: Ctx) -> int:
    cfg = ctx.cfg
    if cfg.stress_drop_pct <= 0:
        raise CliError("the stress rule is off (stress_drop_pct = 0 in config.toml)")
    if cfg.margin_mode != "cross":
        a, m = cfg.stress_drop_pct / 100, cfg.maintenance_margin_rate
        raise CliError(
            f"headroom pools your balance across positions, which only applies to cross margin. With isolated margin "
            f"spare balance does not protect a position: keep each position's leverage <= "
            f"{risk.max_leverage_for_distance(args.side, a, m, 1.0):.1f}x to survive a {a:.0%} move, "
            f"or set margin_mode = \"cross\" if your balance backs every position."
        )
    equity = _equity(args, ctx)
    price = args.price if args.price is not None else _provider(args, ctx).spot()
    holdings = ctx.journal.open_positions()
    total = risk.total_equity(equity, holdings, price)
    if total <= 0:
        raise CliError("equity plus open PnL is not positive at this price")
    net, gross = risk.exposure(holdings, price)
    a, m = cfg.stress_drop_pct / 100, cfg.maintenance_margin_rate
    s = 1 if args.side == "long" else -1

    print(f"ETHUSD {usd(price)}   equity {usd(equity)} + open PnL {signed_usd(total - equity)} = {usd(total)}")
    print("-" * 78)
    print(f"Holdings     gross {usd(gross, 0)} = {gross / total:.2f}x equity, net {signed_usd(net)}")
    for move in (-a, a):
        st = risk.stress_at(total, net, gross, move, m)
        print(f"  {move:+.0%}: equity {usd(st.equity_after, 0)}, {'loss' if st.loss_pct >= 0 else 'gain'} {pct(abs(st.loss_pct), 0)} of equity, margin limit {'holds' if st.survives else 'BREACHED'}")
    room = risk.max_additional_notional(cfg, total, net, gross, args.side)
    print(f"Room         up to {usd(room, 0)} more {args.side} notional = {room / price:.4f} ETH at {usd(price)}")
    full = risk.stress_at(total, net + s * room, gross + room, -s * a, m)
    print(f"  using all of it, a {a:.0%} {'fall' if s == 1 else 'rise'} leaves {usd(full.equity_after, 0)}: {pct(full.loss_pct, 0)} of equity lost")
    liq = risk.account_liquidation_price(total, holdings, price, m)
    if liq is not None and liq <= 0:
        print("Whole-book liquidation: none. Equity exceeds exposure, so the book cannot be liquidated at this size.")
    elif liq is not None:
        print(f"Whole-book liquidation today (cross margin): {usd(liq)} ({pct(abs(1 - liq / price))} from here)")
    print("Fees, funding, basis and venue liquidation fees are not included, so real room is smaller.")
    return 0


def cmd_study(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    df = provider.candles(args.days)
    horizon = args.horizon or ctx.cfg.max_holding_days
    risk_pct = ctx.cfg.risk_per_trade_pct if args.risk_pct is None else args.risk_pct

    def frac(value):
        return None if value is None else value / 100

    trades = study.run(
        ctx.cfg, df, args.side, lookback=args.lookback, trigger=args.trigger / 100, horizon=horizon,
        leverage=args.leverage, stop_pct=frac(args.stop_pct), stop_atr=args.stop_atr,
        target_pct=frac(args.target_pct), target_r=args.target_r,
    )
    anchor = "below its" if args.side == "long" else "above its"
    extreme = "high" if args.side == "long" else "low"
    stop = f"{args.stop_pct:g}%" if args.stop_pct is not None else f"{args.stop_atr:g}x ATR"
    target = "none" if args.target_pct is None and args.target_r is None else (
        f"{args.target_pct:g}%" if args.target_pct is not None else f"{args.target_r:g}R")
    print(f"Rule: {args.side} at the close when price is >= {args.trigger:g}% {anchor} {args.lookback}d {extreme}; "
          f"stop {stop}, target {target}, {args.leverage:g}x, time stop {horizon}d")
    print(f"Data: {len(df)} daily candles, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({provider.name})")
    print("-" * 78)

    summary = study.summarize(trades)
    if summary is None:
        print("No trades triggered in this history.")
        return 0

    shown = trades if len(trades) <= 40 else trades[-20:]
    if len(shown) < len(trades):
        print(f"(showing the last {len(shown)} of {len(trades)} trades)")
    print(f"  {'entered':<11} {'exited':<11} {'entry':>10} {'exit':>10} {'outcome':<10} {'days':>4} {'R':>7}")
    for t in shown:
        print(f"  {t.entry_date:%Y-%m-%d}  {t.exit_date:%Y-%m-%d} {t.entry:>10,.2f} {t.exit:>10,.2f} {t.outcome:<10} {t.days_held:>4} {t.r:>+7.2f}")
    print("-" * 78)
    mix = ", ".join(f"{k} {v}" for k, v in sorted(summary.outcomes.items()))
    print(f"{summary.n} trades, {summary.wins} winners ({summary.wins / summary.n:.0%}); {mix}; avg hold {summary.avg_days:.1f}d")
    print(f"R multiples: avg {summary.avg_r:+.2f}, median {summary.median_r:+.2f}, total {summary.total_r:+.1f}, best {summary.best_r:+.2f}, worst {summary.worst_r:+.2f}")
    total, drawdown = study.compound(trades, risk_pct)
    held = float(df["close"].iloc[-1] / df["close"].iloc[args.lookback - 1] - 1)
    print(f"Risking {risk_pct:g}% per trade: {total:+.1%} total, max drawdown {pct(drawdown)}.  Buy-and-hold ETH over the same window: {held:+.1%}")
    print()
    if summary.n < 30:
        print(f"Only {summary.n} trades: win rate and expectancy are not statistically reliable at this sample size.")
    print("Not tuned. Varying --trigger, --stop or --target until the result looks good fits noise, not edge.")
    print("Daily bars: fills at stop/target are assumed exact (no slippage or gap-through); a same-day stop and target counts as the stop.")
    return 0


def cmd_watch(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    prev = start = df = None
    candles_at = float("-inf")
    last_action: dict[int, str] = {}
    seen: set[tuple[int, str]] = set()
    try:
        while True:
            try:
                price = provider.spot()
            except DataError as exc:
                if prev is None:
                    raise
                print(f"warning: {exc}", file=sys.stderr)
                time.sleep(args.interval)
                continue
            start = price if start is None else start
            now = datetime.now(timezone.utc)
            print(f"{now:%Y-%m-%d %H:%M:%S}Z  ETHUSD {usd(price)}  ({price / start - 1:+.2%} since start)")
            fired = False
            for level in args.above or []:
                if alerts.crossed(prev, price, level):
                    print(f"  ALERT price crossed above {usd(level)}" if price >= level else f"  ALERT price crossed below {usd(level)}")
                    fired = True
            for level in args.below or []:
                if alerts.crossed(prev, price, level):
                    print(f"  ALERT price crossed below {usd(level)}" if price <= level else f"  ALERT price crossed above {usd(level)}")
                    fired = True
            if args.record:
                _record_derivs(ctx)
            statuses =[risk.mark_position(ctx.cfg, p, price, now) for p in ctx.journal.open_positions()]
            results = []
            if statuses:
                if df is None or time.monotonic() - candles_at > CANDLE_REFRESH_SECONDS:
                    candles_at = time.monotonic()
                    try:
                        df = provider.candles(args.days)
                    except DataError as exc:
                        print(f"warning: candles unavailable, no recommendations this poll ({exc})", file=sys.stderr)
                if df is not None:
                    results, _ = _assess_all(args, ctx, provider, price, df)
            actions = {s.position.id: a for s, a in results}
            for s in statuses:
                p = s.position
                liq = f"  liq {pct(s.dist_to_liq)} away" if ctx.cfg.margin_mode == "isolated" else ""
                action = f"  -> {actions[p.id].action}" if p.id in actions else ""
                print(f"  #{p.id} {p.side} {signed_usd(s.unrealized)}{liq}  stop {pct(s.dist_to_stop)} away  held {s.days_held:.1f}d{action}")
            for s, a in results:
                before = last_action.get(s.position.id)
                if before != a.action:
                    if a.action != "HOLD" or before is not None:  # a first sighting of HOLD is not news
                        print(f"  ACTION CHANGE #{s.position.id}: {before or 'new'} -> {a.action}")
                        print_assessment(s, a, ctx.cfg.max_holding_days)
                        fired = True
                    last_action[s.position.id] = a.action
            current = {(s.position.id, f) for s in statuses for f in s.flags}
            if statuses and ctx.journal.equity is not None:
                port = risk.portfolio_status(ctx.cfg, ctx.journal.equity, ctx.journal.peak_equity, statuses)
                current |= {(0, f) for f in port.flags}  # id 0 = the whole book
            for pid, flag in sorted(alerts.new_flags(seen, current)):
                print(f"  ALERT {'portfolio' if pid == 0 else f'#{pid}'} {flag}")
                fired = True
            if fired:
                print("\a", end="", flush=True)
            prev = price
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


def cmd_size(args, ctx: Ctx) -> int:
    equity = _equity(args, ctx)
    provider = None
    entry = args.entry
    if entry is None or not args.no_history:
        provider = _provider(args, ctx)
    if entry is None:
        entry = provider.spot()
    df = None if args.no_history else _fetch_candles(provider, args.history_days)
    now = datetime.now(timezone.utc)
    open_risk = _open_risk(ctx, entry, now)
    plan = risk.plan_position(
        ctx.cfg, equity, args.side, entry, args.stop, args.leverage, target=args.target,
        risk_pct=args.risk_pct, hold_days=args.hold_days, open_risk_usd=open_risk, atr_pct=_atr_pct(df),
        holdings=ctx.journal.open_positions(),
    )
    cross = ctx.cfg.margin_mode == "cross"
    print_plan(plan, equity, cross)
    print()
    if cross:
        print("Cross margin: the exchange leverage setting does not move your liquidation price; the whole-book stress above does.")
    else:
        print_ladder(ctx.cfg, equity, args, entry, open_risk, ctx.journal.open_positions())
    print()
    print("Price scenarios (move from entry):")
    for s in risk.scenarios(plan, equity):
        tag = "  LIQUIDATED" if s.liquidated else ""
        print(f"  {s.move:+4.0%}  {usd(s.price):>12}  {signed_usd(s.pnl):>12}  {s.pnl_pct_equity:+7.1%} of equity{tag}")
    if df is not None:
        print()
        for line in history_lines(df, args.side, ctx.cfg.max_holding_days, plan.stop_dist, plan.liq_dist):
            print(line)
    print()
    print_verdict(plan)
    return 0 if plan.ok else 1


def cmd_open(args, ctx: Ctx) -> int:
    equity = _equity(args, ctx)
    open_risk = _open_risk(ctx, args.entry, datetime.now(timezone.utc))
    df = None if args.no_history else _fetch_candles(_provider(args, ctx), 60)
    plan = risk.plan_position(
        ctx.cfg, equity, args.side, args.entry, args.stop, args.leverage, target=args.target,
        risk_pct=args.risk_pct, qty=args.qty, open_risk_usd=open_risk, atr_pct=_atr_pct(df),
        holdings=ctx.journal.open_positions(),
    )
    print_plan(plan, equity, ctx.cfg.margin_mode == "cross")
    print_verdict(plan)
    if not plan.ok and not args.force:
        print("Not recorded. Fix the plan or pass --force to record it anyway.")
        return 1
    note = args.note or ""
    if not plan.ok:
        note = f"{note} [FORCED past: {'; '.join(plan.violations)}]".strip()
    pos = ctx.journal.open(
        side=args.side, entry=args.entry, qty=plan.qty, leverage=args.leverage, stop=args.stop,
        target=args.target, mmr=ctx.cfg.maintenance_margin_rate, initial_risk_usd=plan.risk_usd, note=note,
    )
    ctx.journal.save()
    print(f"recorded position #{pos.id}")
    return 0


def cmd_close(args, ctx: Ctx) -> int:
    try:
        pos = ctx.journal.get(args.id)
    except KeyError as exc:
        raise CliError(str(exc.args[0])) from exc
    now = datetime.now(timezone.utc)
    status = risk.mark_position(ctx.cfg, pos, args.price, now)
    if "LIQUIDATED" in status.flags:
        gross, fees, funding = -pos.margin, 0.0, 0.0
        realized = gross
        print("exit price is beyond liquidation: recording the loss as the full margin")
    else:
        gross = status.unrealized
        fees = args.fees if args.fees is not None else ctx.cfg.taker_fee_rate * pos.qty * (pos.entry + args.price)
        funding = args.funding if args.funding is not None else status.funding_accrued
        realized = gross - fees - funding
    try:
        ctx.journal.close(pos.id, args.price, realized)
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    ctx.journal.save()
    r = realized / pos.initial_risk_usd if pos.initial_risk_usd else float("nan")
    print(f"closed #{pos.id}: gross {signed_usd(gross)}, fees {usd(fees)}, funding {usd(funding)} -> realized {signed_usd(realized)} ({r:+.2f}R) over {status.days_held:.1f}d")
    print(f"equity now {usd(ctx.journal.equity)}" if ctx.journal.equity is not None else "equity not set; balance not updated")
    return 0


def cmd_risk(args, ctx: Ctx) -> int:
    equity = _equity(args, ctx)
    positions = ctx.journal.open_positions()
    if not positions:
        print(f"No open positions. Equity {usd(equity)}.")
        return 0
    price = args.price if args.price is not None else _provider(args, ctx).spot()
    now = datetime.now(timezone.utc)
    statuses = [risk.mark_position(ctx.cfg, p, price, now) for p in positions]
    print(f"ETHUSD {usd(price)}")
    print("-" * 78)
    for s in statuses:
        p = s.position
        target = f"  target {usd(p.target)}" if p.target else ""
        print(f"#{p.id} {p.side.upper()} {p.qty:.4f} ETH @ {usd(p.entry)}  {p.leverage:g}x  stop {usd(p.stop)}{target}  held {s.days_held:.1f}/{ctx.cfg.max_holding_days}d")
        liq = f"   liq {usd(s.liq_price)} ({pct(s.dist_to_liq)} away)" if ctx.cfg.margin_mode == "isolated" else ""
        print(f"   uPnL {signed_usd(s.unrealized)} ({s.unrealized / p.margin:+.1%} of margin){liq}   stop {pct(s.dist_to_stop)} away   funding ~{usd(s.funding_accrued)}")
        print(f"   loss to stop from here {usd(s.loss_to_stop_usd)}" + (f"   FLAGS: {', '.join(s.flags)}" if s.flags else ""))
        if p.note:
            print(f"   note: {p.note}")
    port = risk.portfolio_status(ctx.cfg, equity, ctx.journal.peak_equity, statuses)
    print("-" * 78)
    print(f"Equity {usd(port.equity)} + uPnL {signed_usd(port.unrealized)} = {usd(port.total_equity)}   drawdown from peak {pct(port.drawdown)}")
    print(f"Margin {usd(port.margin_used)} ({pct(port.margin_utilisation)} of equity)   gross {usd(port.gross_notional)} = {port.effective_leverage:.2f}x   net {signed_usd(port.net_notional)}")
    print(f"Open risk to stops {usd(port.open_risk_usd)} = {pct(port.open_risk_pct, 2)} of equity (limit {ctx.cfg.max_open_risk_pct:g}%)")
    if ctx.cfg.stress_drop_pct > 0:
        verdict = "holds" if port.stress_survives else "BREACHED"
        room = f"; room to add {usd(port.headroom_notional, 0)} notional" if ctx.cfg.margin_mode == "cross" else ""
        print(f"Stress {ctx.cfg.stress_drop_pct:g}%: worst case loses {pct(port.stress_loss_pct, 0)} of equity, margin limit {verdict}{room}")
        if port.account_liq_price is not None and port.account_liq_price <= 0:
            print("Whole-book liquidation: none. Equity exceeds exposure, so the book cannot be liquidated at this size.")
        elif port.account_liq_price is not None:
            print(f"Whole-book liquidation (cross margin): {usd(port.account_liq_price)} ({pct(abs(1 - port.account_liq_price / price))} from here)")
    flags = list(port.flags) + [f"#{s.position.id} {f}" for s in statuses for f in s.flags]
    if flags:
        print(f"FLAGS: {', '.join(flags)}")
    return 1 if flags else 0


def cmd_history(args, ctx: Ctx) -> int:
    closed = ctx.journal.closed_positions()
    if not closed:
        print("No closed positions.")
        return 0
    print(f"{'id':>3} {'side':<5} {'entry':>10} {'exit':>10} {'realized':>12} {'R':>6}  closed")
    for p in closed:
        r = p.realized_pnl / p.initial_risk_usd if p.initial_risk_usd else float("nan")
        print(f"{p.id:>3} {p.side:<5} {p.entry:>10,.2f} {p.exit_price:>10,.2f} {signed_usd(p.realized_pnl):>12} {r:>+6.2f}  {p.closed_at}")
    wins = [p for p in closed if p.realized_pnl > 0]
    total = sum(p.realized_pnl for p in closed)
    print(f"{len(closed)} trades, {len(wins)} winners ({len(wins) / len(closed):.0%}), total {signed_usd(total)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ethtracker", description="ETH-USD tracker, position sizing and risk journal")
    p.add_argument("--provider", choices=["coinbase", "kraken", "csv"], help="price source (default: config)")
    p.add_argument("--csv", type=Path, help="daily candles file for --provider csv")
    p.add_argument("--no-cache", action="store_true", help="bypass the local candle cache")
    p.add_argument("--derivs-csv", type=Path, help="open interest / funding / liquidation history (default: what `record` has collected)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help):
        sp = sub.add_parser(name, help=help)
        sp.set_defaults(fn=fn)
        return sp

    sp = add("init", cmd_init, "write the default config and optionally set equity")
    sp.add_argument("--equity", type=float)
    sp.add_argument("--force", action="store_true", help="overwrite an existing config.toml")

    sp = add("equity", cmd_equity, "show or set account equity (wallet balance, excluding open PnL)")
    sp.add_argument("value", type=float, nargs="?")

    add("price", cmd_price, "print the spot price")

    sp = add("plan", cmd_plan, "set or show the adding plan: a liquidation floor and a weekly add")
    sp.add_argument("--floor", type=float, help="the highest your liquidation price may go, in dollars")
    sp.add_argument("--weekly", type=float, help="new cash added each week, in dollars")
    sp.add_argument("--mmr", type=float, help="your venue's maintenance margin as a fraction (a new plan assumes 0.02)")
    sp.add_argument("--price", type=float, help="price to evaluate at (default: live)")

    sp = add("add", cmd_add, "record new cash and the ETH it bought, sized to keep liquidation at or below the floor")
    sp.add_argument("--usd", type=float, help="new cash added (default: the plan's weekly add; 0 to use existing room)")
    sp.add_argument("--price", type=float, help="your fill price (default: live)")
    sp.add_argument("--qty", type=float, help="ETH actually bought (default: the most the floor allows)")
    sp.add_argument("--force", action="store_true", help="record a size beyond the floor anyway")
    sp.add_argument("--at", help="when you bought, if not now: 2026-10-25 or 2026-10-25T14:30 (UTC unless it has an offset)")
    sp.add_argument("--note", help="why you bought: kept with the buy and the signals at the time")

    sp = add("review", cmd_review, "your buys against what the signals said when you made them, and how each turned out")
    sp.add_argument("--backfill", action="store_true", help="capture the signals for buys recorded without them")

    sp = add("context", cmd_context, "the dashboard's market panels: trend, crowding, signal odds, order fill rates, floor risk")
    sp.add_argument("--floor", type=float, help="also show the risk of reaching this price (keep it out of public logs)")

    sp = add("sync", cmd_sync, "download daily candles into the local cache (later runs fetch only the gap)")
    sp.add_argument("--days", type=int, default=3000)

    sp = add("export", cmd_export, "write candles plus SMA 20/50/100/200, RSI, ATR, volatility and volume average to a CSV")
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument("--days", type=int, default=3000)

    sp = add("headroom", cmd_headroom, "how much more you can add and still survive the stress move, from current equity and holdings")
    sp.add_argument("--side", choices=["long", "short"], default="long")
    sp.add_argument("--price", type=float, help="use this price instead of fetching spot")
    sp.add_argument("--equity", type=float)

    sp = add("snapshot", cmd_snapshot, "price, trend, volatility and historical adverse-move stats")
    sp.add_argument("--days", type=int, default=1000, help="daily candles to fetch (Kraken caps at 720)")

    sp = add("study", cmd_study, "simulate your pullback rule on daily candles (R-multiples, one position at a time)")
    sp.add_argument("--side", choices=["long", "short"], default="long")
    sp.add_argument("--lookback", type=int, default=30, help="days for the reference high (long) or low (short)")
    sp.add_argument("--trigger", type=float, default=15, help="percent below the lookback high (long) or above the lookback low (short)")
    stop = sp.add_mutually_exclusive_group(required=True)
    stop.add_argument("--stop-pct", type=float, help="stop distance, percent of entry")
    stop.add_argument("--stop-atr", type=float, help="stop distance as a multiple of the 14d ATR at entry")
    target = sp.add_mutually_exclusive_group()
    target.add_argument("--target-pct", type=float, help="target distance, percent of entry")
    target.add_argument("--target-r", type=float, help="target as a multiple of the stop distance")
    sp.add_argument("--leverage", type=float, required=True,
                    help="exchange leverage if isolated margin; notional / equity if cross margin")
    sp.add_argument("--horizon", type=int, help="time stop in days (default: config max_holding_days)")
    sp.add_argument("--risk-pct", type=float, help="risk per trade for the compounded figure (default: config)")
    sp.add_argument("--days", type=int, default=3000, help="daily candles to fetch")

    sp = add("signals", cmd_signals, "technical overlays (trend, RSI, volatility, anchored VWAP, volume profile) and derivatives positioning")
    sp.add_argument("--days", type=int, default=500, help="daily candles to use (the volume profile looks back 365)")
    sp.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for the anchored VWAP; default: the 365d low")

    sp = add("advise", cmd_advise, "per open position: recommended action with the bull and bear case (exit 1 on EXIT/TAKE_PARTIAL)")
    sp.add_argument("--days", type=int, default=500)
    sp.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for the anchored VWAP")
    sp.add_argument("--price", type=float, help="use this price instead of fetching spot")

    sp = add("record", cmd_record, "poll Deribit's public ticker and append open interest and funding to a local history")
    sp.add_argument("--instrument", help="default: config derivs_instrument")
    sp.add_argument("--interval", type=float, default=900, help="seconds between snapshots")
    sp.add_argument("--once", action="store_true")

    sp = add("dashboard-json", cmd_dashboard_json, "print (or write) the JSON snapshot the dashboard displays")
    sp.add_argument("--out", type=Path, help="write to this file instead of printing")
    sp.add_argument("--days", type=int, default=500)
    sp.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for the anchored VWAP")
    sp.add_argument("--price", type=float, help="use this price instead of fetching spot")
    sp.add_argument("--basis", choices=snapshot.BASES, default="close", help="close: signals on completed daily candles (default); live: include today's")

    sp = add("publish", cmd_publish, "compute the snapshot, send it to your Vercel dashboard and push your journal (token from ETHTRACKER_INGEST_TOKEN)")
    sp.add_argument("--url", help="dashboard address (default: $ETHTRACKER_DASHBOARD_URL)")
    sp.add_argument("--interval", type=float, default=60, help="seconds between publishes")
    sp.add_argument("--once", action="store_true", help="publish once and exit (run this after you open or close a trade)")
    sp.add_argument("--basis", choices=snapshot.BASES, default="close", help="close: signals on completed daily candles (default); live: include today's")
    sp.add_argument("--days", type=int, default=500)
    sp.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for the anchored VWAP")

    sp = add("cloud-run", cmd_cloud_run, "the daily job: pull your journal and derivatives history from the dashboard, add today's data, publish")
    sp.add_argument("--url", help="dashboard address (default: $ETHTRACKER_DASHBOARD_URL)")
    sp.add_argument("--dry-run", action="store_true", help="fetch live data and compute, but publish nothing (needs no dashboard)")
    sp.add_argument("--out", type=Path, help="also write the snapshot JSON here")
    sp.add_argument("--days", type=int, default=500)

    sp = add("swing-study", cmd_swing_study, "buy a sharp pullback, sell only into a spike, on ETH or BTC, versus holding spot")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")

    sp = add("timing-study", cmd_timing_study, "buy weekly but pick the day within a few days; and how often price broke a floor")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")
    sp.add_argument("--floor", type=float, help="your liquidation floor, in dollars")

    sp = add("score-study", cmd_score_study, "check the signal score's parts and buckets against what followed, by era")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")
    sp.add_argument("--horizon", type=int, default=score.HORIZON, help="days ahead to judge each reading on")

    add("account-check", cmd_account_check, "which Coinbase endpoints your view-only key can read (shapes only, no values)")

    sp = add("cap-study", cmd_cap_study, "the floor plan with new cash's leverage capped, fixed or by the signal score")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")
    sp.add_argument("--years", type=int, default=cycles.HORIZON_YEARS, help="window length in years")
    sp.add_argument("--since", help="only windows starting on or after this date, e.g. 2021-01-01")

    sp = add("dip-study", cmd_dip_study, "add cash weekly, or save it for 20% dips, at 1x, 2.5x, 3x and under the floor plan")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")
    sp.add_argument("--years", type=int, default=cycles.HORIZON_YEARS, help="window length in years")
    sp.add_argument("--since", help="only windows starting on or after this date, e.g. 2021-01-01")

    sp = add("entry-study", cmd_entry_study, "put cash to work now or bid lower, when each buy is sized to a liquidation floor")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")
    sp.add_argument("--under", type=float, help="also a floor this far under the entry close, as a fraction (0.47)")

    sp = add("pullback-study", cmd_pullback_study, "how often 'the pullback is over' signals were right, buy orders under the price, and weekly adding")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")

    sp = add("trim-study", cmd_trim_study, "trim into spikes, refill on dips, liquidation pinned to one price, on ETH or BTC")
    sp.add_argument("--asset", choices=["ETH", "BTC", "eth", "btc"], default="ETH")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch")

    sp = add("short-study", cmd_short_study, "short-term shorts on overheating signals, alone and as a cut to the leveraged core")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch (Coinbase ETH-USD starts in May 2016)")

    sp = add("combo-study", cmd_combo_study, "a long-only perp core plus a small high-leverage swing sleeve feeding it, versus spot")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch (Coinbase ETH-USD starts in May 2016)")

    sp = add("cycle-study", cmd_cycle_study, "dip-buying with leverage versus holding spot, inside each bull phase and over the whole history")
    sp.add_argument("--days", type=int, default=4000, help="daily candles to fetch (Coinbase ETH-USD starts in May 2016)")

    sp = add("signals-study", cmd_signals_study, "how the technical risk-flag count related to later drawdowns and rallies in history")
    sp.add_argument("--horizon", type=int, help="days ahead (default: config max_holding_days)")
    sp.add_argument("--days", type=int, default=3000)

    sp = add("watch", cmd_watch, "poll the price, check level alerts and open-position flags")
    sp.add_argument("--interval", type=float, default=60, help="seconds between polls")
    sp.add_argument("--above", type=float, action="append", help="alert when price crosses this level (repeatable)")
    sp.add_argument("--below", type=float, action="append", help="alert when price crosses this level (repeatable)")
    sp.add_argument("--once", action="store_true", help="check once and exit (for cron)")
    sp.add_argument("--record", action="store_true", help="also append a Deribit open-interest/funding snapshot each poll")
    sp.add_argument("--days", type=int, default=500, help="daily candles used for the signals")
    sp.add_argument("--anchor", help="anchor date (YYYY-MM-DD) for the anchored VWAP; default: the 365d low")

    def trade_args(sp, entry_required: bool):
        sp.add_argument("--side", choices=["long", "short"], required=True)
        sp.add_argument("--entry", type=float, required=entry_required, help="default: live spot" if not entry_required else None)
        sp.add_argument("--stop", type=float, required=True)
        sp.add_argument("--target", type=float)
        sp.add_argument("--leverage", type=float, required=True)
        sp.add_argument("--risk-pct", type=float, help="risk budget, % of equity (default: config)")
        sp.add_argument("--equity", type=float, help="override recorded equity")

    sp = add("size", cmd_size, "size a trade from its stop and test it against the risk rules (exit 1 if blocked)")
    trade_args(sp, entry_required=False)
    sp.add_argument("--hold-days", type=float, help="funding horizon (default: config max_holding_days)")
    sp.add_argument("--history-days", type=int, default=1000)
    sp.add_argument("--no-history", action="store_true", help="skip the candle download (skips the stop-vs-ATR check)")

    sp = add("open", cmd_open, "record a position you have taken (rejects rule-breaking plans unless --force)")
    trade_args(sp, entry_required=True)
    sp.add_argument("--qty", type=float, help="actual ETH quantity filled (default: risk-sized)")
    sp.add_argument("--note")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-history", action="store_true", help="skip the candle download (skips the stop-vs-ATR check)")

    sp = add("close", cmd_close, "close a recorded position and update equity")
    sp.add_argument("id", type=int)
    sp.add_argument("--price", type=float, required=True)
    sp.add_argument("--fees", type=float, help="actual USD fees (default: estimate)")
    sp.add_argument("--funding", type=float, help="actual USD funding paid, negative if received (default: estimate)")

    sp = add("risk", cmd_risk, "mark open positions to market and check portfolio risk (exit 1 if any flag)")
    sp.add_argument("--price", type=float, help="mark at this price instead of fetching")
    sp.add_argument("--equity", type=float)

    add("history", cmd_history, "closed trades and realized results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    home = config.home()
    try:
        journal = Journal(home / "state.json")
        ctx = Ctx(cfg=plan.with_plan_margin(config.load(home / "config.toml"), journal.plan), journal=journal, home=home)
        return args.fn(args, ctx)
    except (CliError, DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
