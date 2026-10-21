"""The dashboard's data: one JSON snapshot of the book and the market, and the call that publishes it.

Everything is computed here, on your machine, by the same functions behind ``signals`` and ``advise``,
so the dashboard cannot disagree with the CLI. The hosted dashboard only stores and displays the result:
it never sees an exchange key and never runs the maths.

The snapshot contains your equity and positions. It is sent over HTTPS with a bearer token and stored in
the dashboard's Redis, so treat the dashboard as holding that data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
import requests

from . import advice, indicators, risk, signals
from .config import Config
from .data import DataError
from .journal import Journal

VERSION = 1
RECENT_CLOSED = 5
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _clean(value):
    """JSON has no NaN or infinity: map them to None, recursing through containers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


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
) -> dict:
    now = now or datetime.now(timezone.utc)
    priced = indicators.with_price(df, price)
    sigs = signals.compute(cfg, priced, derivs, anchor)
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

    positions = []
    for status in statuses:
        p = status.position
        a = advice.assess(cfg, p, status, sigs, priced, book_flags)
        positions.append({
            "id": p.id, "side": p.side, "qty": p.qty, "entry": p.entry, "leverage": p.leverage,
            "stop": p.stop, "target": p.target, "opened_at": p.opened_at, "note": p.note,
            "held_days": status.days_held, "max_days": cfg.max_holding_days,
            "mark": status.price, "move": a.move, "unrealized": status.unrealized,
            "dist_to_stop": status.dist_to_stop, "funding_accrued": status.funding_accrued,
            "liq_price": status.liq_price if isolated else None,
            "dist_to_liq": status.dist_to_liq if isolated else None,
            "flags": list(status.flags), "action": a.action, "reasons": list(a.reasons),
            "bull": list(a.bull_case), "bear": list(a.bear_case),
            "suggested_stop": a.suggested_stop, "suggested_fraction": a.suggested_fraction,
            "net_score": a.net_score, "risk_flags": a.risk_flags, "deep_itm": a.deep_itm,
        })

    closed = [
        {
            "id": p.id, "side": p.side, "entry": p.entry, "exit": p.exit_price, "closed_at": p.closed_at,
            "realized_pnl": p.realized_pnl,
            "r": p.realized_pnl / p.initial_risk_usd if p.initial_risk_usd else None,
        }
        for p in journal.closed_positions()[-RECENT_CLOSED:][::-1]
    ]

    snap = {
        "version": VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "price": float(price),
        "price_1d": float(price / df["close"].iloc[-2] - 1) if len(df) >= 2 else None,
        "source": source,
        "equity": equity_block,
        "portfolio": portfolio,
        "positions": positions,
        "closed": closed,
        "signals": [
            {"name": g.name, "family": g.family or g.name, "bias": g.bias, "risk": g.risk,
             "headline": g.headline, "unvalidated": g.unvalidated}
            for g in sigs.signals
        ],
        "votes": sigs.votes(),
        "levels": {key: float(value) for key, value in sigs.levels.items()},
        "notes": list(sigs.notes),
        "config": {
            "margin_mode": cfg.margin_mode, "max_holding_days": cfg.max_holding_days,
            "max_open_risk_pct": cfg.max_open_risk_pct,
            "deep_itm_pct": cfg.deep_itm_pct, "exit_risk_flags": cfg.exit_risk_flags,
            "stress_drop_pct": cfg.stress_drop_pct, "funding_hot_8h": cfg.funding_hot_8h,
            "instrument": cfg.derivs_instrument,
        },
    }
    return _clean(snap)


def ingest_endpoint(base: str) -> str:
    """The ingest URL for a dashboard base URL. The token travels in a header, so plain http is refused off localhost."""
    parsed = urlparse(base if "//" in base else f"https://{base}")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS):
        raise DataError(f"refusing to send a token over {parsed.scheme}://: use https (http is allowed only for localhost)")
    if not parsed.hostname:
        raise DataError(f"not a usable dashboard URL: {base!r}")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/ingest"):
        path += "/api/ingest"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def publish(endpoint: str, token: str, snap: dict) -> None:
    """POST the snapshot. Errors name the status and a body excerpt, never the token."""
    try:
        response = requests.post(
            endpoint, json=snap, headers={"Authorization": f"Bearer {token}"}, timeout=20,
        )
    except requests.RequestException as exc:
        raise DataError(f"could not reach {endpoint}: {exc.__class__.__name__}") from exc
    if not response.ok:
        raise DataError(f"dashboard refused the snapshot ({response.status_code}): {response.text[:200]!r}")
