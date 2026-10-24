"""Command-line interface: ``ethtracker <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import advice, alerts, config, cycles, derivs, indicators, risk, signals, snapshot, study
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


def _provider(args, ctx: Ctx) -> Provider:
    """The configured source; network sources go through the local candle cache unless --no-cache."""
    provider = get_provider(args.provider or ctx.cfg.provider, args.csv)
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


DERIVS_HISTORY_ROWS = 2000  # one row a day from the daily job: years of history, bounded


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
    snap = _snapshot(args, ctx, provider, price, provider.candles(args.days))
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
    refresh = snapshot.DAY_SECONDS if args.once else int(args.interval)  # after --once, the daily job is the cadence
    df, candles_at, first = None, float("-inf"), True
    try:
        while True:
            try:
                price = provider.spot()
                if df is None or time.monotonic() - candles_at > CANDLE_REFRESH_SECONDS:
                    df, candles_at = provider.candles(args.days), time.monotonic()
                history = None if args.derivs_csv else _derivs_from_text(snapshot.get_state(base, token, "derivs"))
                snap = _snapshot(args, ctx, provider, price, df, history, refresh_seconds=refresh, publisher="local")
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
        snap = snapshot.build(ctx.cfg, journal, df, price, derivs_df, source=provider.name, basis="close",
                              refresh_seconds=snapshot.DAY_SECONDS, publisher=publisher)
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
            snapshot.put_state(base, token, "derivs", snapshot.tail_csv(history_path.read_text(), DERIVS_HISTORY_ROWS))
        print("published the snapshot" + (" and the derivatives history" if recorded else ""))
    return 0


def _cycle_table(title: str, rows: list) -> None:
    spot = rows[0].multiple
    print()
    print(title)
    print(f"  {'strategy':<40} {'x start':>12} {'vs spot':>8} {'max DD':>7} {'trades':>6} {'in mkt':>6}  liquidated")
    for r in rows:
        vs = f"{r.multiple / spot:.2f}x" if spot > 0 else "n/a"
        liq = (f"account {r.liquidated:%Y-%m-%d}" if r.liquidated is not None
               else f"{r.lost_positions} position(s)" if r.lost_positions else "")
        print(f"  {r.name:<40} {r.multiple:>11,.2f}x {vs:>8} {r.max_drawdown:>7.0%} {r.trades:>6} {r.exposure:>6.0%}  {liq}")


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
        ctx = Ctx(cfg=config.load(home / "config.toml"), journal=Journal(home / "state.json"), home=home)
        return args.fn(args, ctx)
    except (CliError, DataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
