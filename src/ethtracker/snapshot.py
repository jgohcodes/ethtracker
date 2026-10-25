"""The dashboard's data: one JSON snapshot of the book and the market, and the calls that move it.

Everything is computed by the same functions behind ``signals`` and ``advise``, so the dashboard cannot
disagree with the CLI. The hosted dashboard only stores and displays results: it never sees an exchange
key and never runs the maths.

Two producers publish snapshots: the daily job (``cloud-run``, on GitHub Actions) and ``publish`` on your
machine. Both compute signals on **completed daily candles** by default, because you trade the 1D chart
and an unfinished day repaints. Positions are marked at the current spot price.

The dashboard also keeps two pieces of state for the daily job: your journal (pushed by ``publish``) and
the open-interest/funding history the job accumulates. Both live in the dashboard's Redis, never in the
repository. The snapshot contains your equity and positions, so treat the dashboard as holding that data.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests

from . import advice, cycles, decisions, indicators, plan, risk, score, signals
from .config import Config
from .data import DataError
from .journal import Journal

VERSION = 1
RECENT_CLOSED = 5
DAY_SECONDS = 86_400
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
STATE_NAMES = ("journal", "derivs", "trades")
BASES = ("close", "live")


def _clean(value):
    """JSON has no NaN or infinity: map them to None, recursing through containers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _plan_book(cfg: Config, statuses: list, floor: float, price: float, now: datetime) -> tuple[list, list[dict]]:
    """Under an adding plan every weekly buy is its own journal entry, but they are one position: show them
    as one (average entry, whole size, held since the first buy) with each buy listed under it. Shorts, if
    any, stay separate."""
    longs = [st for st in statuses if st.position.side == "long"]
    first = min(longs, key=lambda st: st.position.opened_at).position
    qty = sum(st.position.qty for st in longs)
    cost = sum(st.position.qty * st.position.entry for st in longs)
    margin = sum(st.position.margin for st in longs)
    book = replace(first, entry=cost / qty, qty=qty, leverage=cost / margin, stop=floor, target=None,
                   initial_risk_usd=sum(st.position.initial_risk_usd for st in longs), note=f"{len(longs)} buys")
    status = replace(risk.mark_position(cfg, book, price, now),
                     funding_accrued=sum(st.funding_accrued for st in longs))  # each buy pays from its own date
    lots = [{"id": st.position.id, "opened_at": st.position.opened_at, "qty": st.position.qty,
             "entry": st.position.entry, "unrealized": st.unrealized, "note": st.position.note}
            for st in sorted(longs, key=lambda st: st.position.opened_at)]
    return [status] + [st for st in statuses if st.position.side != "long"], lots


def closed_candles(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Only days that have finished (UTC): today's candle is still forming."""
    return df[df.index < pd.Timestamp(now).tz_convert("UTC").normalize()]


def _with_entry_signals(cfg: Config, account: dict, history: pd.DataFrame, funding8h, derivs) -> dict:
    """Each open trade on the exchange with the score and the signals as they stood when it was made, and each
    position's entry score: its trades' scores weighted by how much of each is still open."""
    account = {**account, "positions": [dict(p) for p in account.get("positions") or []]}
    for p in account["positions"]:
        trades = [dict(t) for t in p.get("trades") or []]
        for t, then in zip(trades, score.at_times(cfg, history, funding8h, derivs, [t.get("time") for t in trades])):
            t["signals"] = then
        scored = [(t["signals"]["score"], abs(t["size"])) for t in trades if t.get("signals") and t["signals"]["score"] is not None]
        weight = sum(w for _, w in scored)
        p["trades"] = trades
        p["entry_score"] = sum(s * w for s, w in scored) / weight if weight else None
    return account


def build(
    cfg: Config,
    journal: Journal,
    df: pd.DataFrame,
    price: float,
    derivs: pd.DataFrame | None = None,
    *,
    now: datetime | None = None,
    anchor: pd.Timestamp | None = None,
    source: str = "",
    basis: str = "close",
    refresh_seconds: int | None = None,
    publisher: str = "local",
    history: pd.DataFrame | None = None,
    funding8h: pd.Series | None = None,
    account: dict | None = None,
) -> dict:
    """Assemble the snapshot. ``basis="close"`` computes signals on completed daily candles only;
    ``"live"`` folds the current price into today's unfinished candle. Positions are marked at ``price``.

    ``history`` (years of daily candles) feeds the trend, the odds of the pullback signals and, under an
    adding plan, the floor risk and the order ladder; without it they use ``df``. ``funding8h`` (about a
    year of daily funding) places today's funding against its own range."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    now = now or datetime.now(timezone.utc)
    closed = closed_candles(df, now)
    if closed.empty:
        raise DataError("no completed daily candles to compute signals from")
    frame = closed if basis == "close" else indicators.with_price(df, price)
    sigs = signals.compute(cfg, frame, derivs, anchor)
    floor = journal.plan["floor"] if journal.plan else None
    if floor is not None:  # the plan's floor replaces the swing rules: no time stop, no stress or risk-per-trade limits
        cfg = replace(plan.with_plan_margin(cfg, journal.plan), max_holding_days=10**9, stress_drop_pct=0.0,
                      max_open_risk_pct=1e9)
    statuses = [risk.mark_position(cfg, p, price, now) for p in journal.open_positions()]
    isolated = cfg.margin_mode == "isolated"

    equity_block = portfolio = None
    book_flags: tuple[str, ...] = ()
    if journal.equity is not None:
        port = risk.portfolio_status(cfg, journal.equity, journal.peak_equity, statuses) if statuses else None
        equity_block = {
            "wallet": journal.equity,
            "unrealized": port.unrealized if port else 0.0,
            "total": port.total_equity if port else journal.equity,
            "peak": journal.peak_equity,
        }
        if port:
            book_flags = port.flags
            portfolio = {
                "margin_used": port.margin_used, "margin_utilisation": port.margin_utilisation,
                "gross_notional": port.gross_notional, "net_notional": port.net_notional,
                "effective_leverage": port.effective_leverage, "open_risk_usd": port.open_risk_usd,
                "open_risk_pct": port.open_risk_pct, "drawdown": port.drawdown,
                "stress": {
                    "drop_pct": cfg.stress_drop_pct, "loss_pct": port.stress_loss_pct,
                    "survives": port.stress_survives, "headroom_notional": port.headroom_notional,
                },
                "account_liq_price": port.account_liq_price, "flags": list(port.flags),
            }

    shown, lots = statuses, None
    if floor is not None and sum(st.position.side == "long" for st in statuses) > 1:
        shown, lots = _plan_book(cfg, statuses, floor, price, now)
    positions = []
    for status in shown:
        p = status.position
        a = advice.assess(cfg, p, status, sigs, frame, book_flags, plan_floor=floor)
        positions.append({
            "id": p.id, "side": p.side, "qty": p.qty, "entry": p.entry, "leverage": p.leverage,
            "stop": p.stop, "target": p.target, "opened_at": p.opened_at, "note": p.note,
            "held_days": status.days_held, "max_days": None if floor is not None else cfg.max_holding_days,
            "mark": status.price, "move": a.move, "unrealized": status.unrealized,
            "dist_to_stop": status.dist_to_stop, "funding_accrued": status.funding_accrued,
            "liq_price": status.liq_price if isolated else None,
            "dist_to_liq": status.dist_to_liq if isolated else None,
            "flags": list(status.flags), "action": a.action, "reasons": list(a.reasons),
            "bull": list(a.bull_case), "bear": list(a.bear_case),
            "suggested_stop": a.suggested_stop, "suggested_fraction": a.suggested_fraction,
            "net_score": a.net_score, "risk_flags": a.risk_flags, "deep_itm": a.deep_itm,
            "lots": lots if lots and p.side == "long" else None,
        })

    closed_trades = [
        {
            "id": p.id, "side": p.side, "entry": p.entry, "exit": p.exit_price, "closed_at": p.closed_at,
            "realized_pnl": p.realized_pnl,
            "r": p.realized_pnl / p.initial_risk_usd if p.initial_risk_usd else None,
        }
        for p in journal.closed_positions()[-RECENT_CLOSED:][::-1]
    ]

    # one venue's flash-crash print never reaches a perp's mark price: the studies drop them, and so does this
    context, _ = cycles.ignore_flash_prints(closed_candles(history, now) if history is not None else closed)
    plan_block = risk_block = None
    if floor is not None:
        longs = [st for st in statuses if st.position.side == "long"]
        total = equity_block["total"] if equity_block else 0.0
        liq = portfolio["account_liq_price"] if portfolio else None
        plan_block = plan.sizing(cfg, journal.plan, float(price), total, sum(st.position.qty for st in longs), liq,
                                 plan.ladder_stats(context))
        risk_block = plan.floor_risk(context, float(price), floor)

    last_close = float(closed["close"].iloc[-1])
    snap = {
        "version": VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "publisher": publisher,
        "refresh_seconds": refresh_seconds,
        "basis": basis,
        "close_date": f"{closed.index[-1]:%Y-%m-%d}",
        "last_close": last_close,
        "price": float(price),
        "price_1d": float(price / last_close - 1),
        "source": source,
        "equity": equity_block,
        "portfolio": portfolio,
        "positions": positions,
        "closed": closed_trades,
        "signals": [
            {"name": g.name, "family": g.family or g.name, "bias": g.bias, "risk": g.risk,
             "headline": g.headline, "unvalidated": g.unvalidated}
            for g in sigs.signals
        ],
        "votes": sigs.votes(),
        "levels": {key: float(value) for key, value in sigs.levels.items()},
        "notes": list(sigs.notes),
        "plan": plan_block,
        "floor_risk": risk_block,
        "trend": plan.trend(context),
        "crowding": plan.crowding(funding8h),
        "odds": plan.odds(context),
        "score": score.block(cfg, context, funding8h, frame, derivs),
        "account": _with_entry_signals(cfg, account, context, funding8h, derivs) if account else None,
        "buys": decisions.review(context, journal.decisions, {p.id: p for p in journal.positions}, float(price))
        if journal.decisions else None,
        "config": {
            "margin_mode": cfg.margin_mode, "max_holding_days": None if floor is not None else cfg.max_holding_days,
            "max_open_risk_pct": cfg.max_open_risk_pct,
            "deep_itm_pct": cfg.deep_itm_pct, "exit_risk_flags": cfg.exit_risk_flags,
            "stress_drop_pct": cfg.stress_drop_pct, "funding_hot_8h": cfg.funding_hot_8h,
            "instrument": cfg.derivs_instrument,
        },
    }
    return _clean(snap)


def public_summary(snap: dict) -> str:
    """One line safe for public CI logs: market facts and signal counts only. Nothing about the account,
    not even how many positions are open, because the repository's logs are public."""
    votes = ", ".join(f"{k} {v:+d}" for k, v in snap["votes"].items() if v) or "none"
    risk_flags = sum(1 for s in snap["signals"] if s["risk"])
    derivs = "yes" if any(s["name"] in ("funding", "oi_change", "oi_level") for s in snap["signals"]) else "no"
    return (f"close {snap['close_date']} at {snap['last_close']:,.2f}; {len(snap['signals'])} signals, "
            f"{risk_flags} risk flag(s), family votes: {votes}; derivatives signals: {derivs}")


# ---- talking to the dashboard ------------------------------------------------------------------------

def dashboard_base(base: str) -> str:
    """Normalise a dashboard address. The token travels in a header, so plain http is refused off localhost."""
    parsed = urlparse(base if "//" in base else f"https://{base}")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS):
        raise DataError(f"refusing to send a token over {parsed.scheme}://: use https (http is allowed only for localhost)")
    if not parsed.hostname:
        raise DataError(f"not a usable dashboard URL: {base!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/api/ingest"):
        path = path[: -len("/api/ingest")]
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def ingest_endpoint(base: str) -> str:
    return dashboard_base(base) + "/api/ingest"


def state_endpoint(base: str, name: str) -> str:
    if name not in STATE_NAMES:
        raise ValueError(f"state name must be one of {STATE_NAMES}")
    return dashboard_base(base) + f"/api/state?name={name}"


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def publish(endpoint: str, token: str, snap: dict) -> None:
    """POST the snapshot. Errors name the status and a body excerpt, never the token."""
    try:
        response = requests.post(endpoint, json=snap, headers=_auth(token), timeout=20)
    except requests.RequestException as exc:
        raise DataError(f"could not reach {endpoint}: {exc.__class__.__name__}") from exc
    if not response.ok:
        raise DataError(f"dashboard refused the snapshot ({response.status_code}): {response.text[:200]!r}")


WRONG_ADDRESS = ("check that ETHTRACKER_DASHBOARD_URL is the site's production address (the domain listed under "
                 "Domains on the Vercel project page), not a deployment link or a vercel.com page")


def _what_came_back(response, url: str) -> str:
    """Status, content type, a redirect off the dashboard's host and a web page's title: enough to tell a login page
    or a wrong address from the API. Never the body, which can hold the journal, so it is safe for public logs."""
    headers = getattr(response, "headers", None) or {}
    kind = (headers.get("content-type") or "").split(";")[0].strip() or "no content type"
    parts = [f"HTTP {response.status_code}", kind]
    final = getattr(response, "url", None)
    if getattr(response, "history", None) and final:
        host = urlparse(final).hostname
        parts.append("redirected on the same host" if host == urlparse(url).hostname else f"redirected to {host}")
    if "html" in kind:
        title = re.search(r"<title[^>]*>(.*?)</title>", (response.text or "")[:20000], re.I | re.S)
        parts.append(f"a web page titled {' '.join(title.group(1).split())[:80]!r}" if title else "a web page")
    return ", ".join(parts)


def get_state(base: str, token: str, name: str) -> str | None:
    """The stored ``journal`` or ``derivs`` text, or None if nothing has been stored yet."""
    url = state_endpoint(base, name)
    try:
        response = requests.get(url, headers=_auth(token), timeout=20)
    except requests.RequestException as exc:
        raise DataError(f"could not reach {url}: {exc.__class__.__name__}") from exc
    if response.status_code == 404:
        return None
    if not response.ok:
        raise DataError(f"dashboard refused to return {name} ({response.status_code}): {response.text[:200]!r}")
    try:
        content = response.json()["content"]
    except (ValueError, KeyError, TypeError):
        content = None
    if not isinstance(content, str):
        raise DataError(f"unexpected {name} payload from the dashboard ({_what_came_back(response, url)}): {WRONG_ADDRESS}")
    return content


def put_state(base: str, token: str, name: str, content: str) -> None:
    url = state_endpoint(base, name)
    try:
        response = requests.put(url, json={"content": content}, headers=_auth(token), timeout=20)
    except requests.RequestException as exc:
        raise DataError(f"could not reach {url}: {exc.__class__.__name__}") from exc
    if not response.ok:
        raise DataError(f"dashboard refused to store {name} ({response.status_code}): {response.text[:200]!r}")


def tail_csv(text: str, rows: int) -> str:
    """Header plus the last ``rows`` lines, so the stored history cannot grow without bound."""
    lines = text.strip().splitlines()
    if len(lines) <= rows + 1:
        return "\n".join(lines) + "\n"
    return "\n".join([lines[0], *lines[-rows:]]) + "\n"
