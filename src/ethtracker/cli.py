"""Command-line interface: ``ethtracker <command>``."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import alerts, config, indicators, risk
from .data import DataError, Provider, get_provider
from .journal import Journal


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


def _provider(args, ctx: Ctx) -> Provider:
    return get_provider(args.provider or ctx.cfg.provider, args.csv)


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


def print_plan(plan: risk.SizePlan, equity: float) -> None:
    arrow = "LONG" if plan.side == "long" else "SHORT"
    target = f"  target {usd(plan.target)}" if plan.target else ""
    print(f"ETHUSD {arrow} @ {plan.leverage:g}x   entry {usd(plan.entry)}  stop {usd(plan.stop)}{target}")
    print("-" * 78)
    print(f"Size         {plan.qty:.4f} ETH   notional {usd(plan.notional)}   margin {usd(plan.margin)} ({pct(plan.margin_pct_equity)} of equity)")
    print(f"Exposure     {plan.effective_leverage:.2f}x equity (notional / equity), equity {usd(equity)}")
    print(f"Risk         loss at stop {usd(plan.risk_usd)} = {pct(plan.risk_pct, 2)} of equity (fees {usd(plan.fees_at_stop_usd)} included)")
    print(f"Liquidation  {usd(plan.liq_price)} ({pct(plan.liq_dist)} from entry); stop is {pct(plan.stop_dist)} away = {pct(plan.stop_fraction_of_liq, 0)} of that distance")
    print(f"Max leverage for this stop: {plan.max_leverage_for_stop:.1f}x")
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


def print_ladder(cfg: config.Config, equity: float, args, entry: float, open_risk: float) -> None:
    print("Leverage ladder for this stop (same risk budget; only margin and liquidation move; liquidation rules only):")
    print(f"  {'lev':>5} {'margin':>12} {'liq price':>12} {'liq dist':>9} {'stop/liq':>9}  verdict")
    for lev in (3, 5, 7, 10, 12, 15):
        p = risk.plan_position(cfg, equity, args.side, entry, args.stop, lev, target=args.target,
                               risk_pct=args.risk_pct, hold_days=args.hold_days, open_risk_usd=open_risk)
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
    print(f"SMA 200      {vs(snap.sma_200)}")
    print(f"RSI 14       {na(snap.rsi_14, lambda x: f'{x:.1f}')}")
    print(f"ATR 14       {na(snap.atr_14, usd)} = {na(snap.atr_14_pct, pct)} of price (typical daily range)")
    print(f"Volatility   30d realised {na(snap.vol_30d, pct)} annualised, typical 1-day move {na(snap.daily_sigma, pct)}")
    print(f"90d range    {usd(snap.low_90d)} - {usd(snap.high_90d)}; {pct(snap.drawdown_from_90d_high)} below the 90d high")
    print()
    for side in ("long", "short"):
        for line in history_lines(df, side, ctx.cfg.max_holding_days):
            print(line)
    return 0


def cmd_watch(args, ctx: Ctx) -> int:
    provider = _provider(args, ctx)
    prev = start = None
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
            statuses = [risk.mark_position(ctx.cfg, p, price, now) for p in ctx.journal.open_positions()]
            for s in statuses:
                p = s.position
                print(f"  #{p.id} {p.side} {signed_usd(s.unrealized)}  liq {pct(s.dist_to_liq)} away  stop {pct(s.dist_to_stop)} away  held {s.days_held:.1f}d")
            current = {(s.position.id, f) for s in statuses for f in s.flags}
            for pid, flag in sorted(alerts.new_flags(seen, current)):
                print(f"  ALERT #{pid} {flag}")
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
    )
    print_plan(plan, equity)
    print()
    print_ladder(ctx.cfg, equity, args, entry, open_risk)
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
    )
    print_plan(plan, equity)
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
        print(f"   uPnL {signed_usd(s.unrealized)} ({s.unrealized / p.margin:+.1%} of margin)   liq {usd(s.liq_price)} ({pct(s.dist_to_liq)} away)   stop {pct(s.dist_to_stop)} away   funding ~{usd(s.funding_accrued)}")
        print(f"   loss to stop from here {usd(s.loss_to_stop_usd)}" + (f"   FLAGS: {', '.join(s.flags)}" if s.flags else ""))
        if p.note:
            print(f"   note: {p.note}")
    port = risk.portfolio_status(ctx.cfg, equity, ctx.journal.peak_equity, statuses)
    print("-" * 78)
    print(f"Equity {usd(port.equity)} + uPnL {signed_usd(port.unrealized)} = {usd(port.total_equity)}   drawdown from peak {pct(port.drawdown)}")
    print(f"Margin {usd(port.margin_used)} ({pct(port.margin_utilisation)} of equity)   gross {usd(port.gross_notional)} = {port.effective_leverage:.2f}x   net {signed_usd(port.net_notional)}")
    print(f"Open risk to stops {usd(port.open_risk_usd)} = {pct(port.open_risk_pct, 2)} of equity (limit {ctx.cfg.max_open_risk_pct:g}%)")
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

    sp = add("snapshot", cmd_snapshot, "price, trend, volatility and historical adverse-move stats")
    sp.add_argument("--days", type=int, default=1000, help="daily candles to fetch (Kraken caps at 720)")

    sp = add("watch", cmd_watch, "poll the price, check level alerts and open-position flags")
    sp.add_argument("--interval", type=float, default=60, help="seconds between polls")
    sp.add_argument("--above", type=float, action="append", help="alert when price crosses this level (repeatable)")
    sp.add_argument("--below", type=float, action="append", help="alert when price crosses this level (repeatable)")
    sp.add_argument("--once", action="store_true", help="check once and exit (for cron)")

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
        ctx = Ctx(cfg=config.load(home / "config.toml"), journal=Journal(home / "state.json"), home=home)
        return args.fn(args, ctx)
    except (CliError, DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
