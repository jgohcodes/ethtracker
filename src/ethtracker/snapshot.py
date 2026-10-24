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
DAY_SECONDS = 86_400
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
STATE_NAMES = ("journal", "derivs")
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


def closed_candles(df: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Only days that have finished (UTC): today's candle is still forming."""
    return df[df.index < pd.Timestamp(now).tz_convert("UTC").normalize()]


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
) -> dict:
    """Assemble the snapshot. ``basis="close"`` computes signals on completed daily candles only;
    ``"live"`` folds the current price into today's unfinished candle. Positions are marked at ``price``."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}")
    now = now or datetime.now(timezone.utc)
    closed = closed_candles(df, now)
    if closed.empty:
        raise DataError("no completed daily candles to compute signals from")
    frame = closed if basis == "close" else indicators.with_price(df, price)
    sigs = signals.compute(cfg, frame, derivs, anchor)
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
        a = advice.assess(cfg, p, status, sigs, frame, book_flags)
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

    closed_trades = [
        {
            "id": p.id, "side": p.side, "entry": p.entry, "exit": p.exit_price, "closed_at": p.closed_at,
            "realized_pnl": p.realized_pnl,
            "r": p.realized_pnl / p.initial_risk_usd if p.initial_risk_usd else None,
        }
        for p in journal.closed_positions()[-RECENT_CLOSED:][::-1]
    ]

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
        "config": {
            "margin_mode": cfg.margin_mode, "max_holding_days": cfg.max_holding_days,
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
    except (ValueError, KeyError, TypeError) as exc:
        raise DataError(f"unexpected {name} payload from the dashboard") from exc
    if not isinstance(content, str):
        raise DataError(f"unexpected {name} payload from the dashboard")
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
