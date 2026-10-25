"""Trades worked out from the exchange's own position, run to run, for a market whose trade history the API
does not give: the Coinbase app's ETH perp is Deribit's ETH_USDC-PERPETUAL, and Coinbase's fills and orders
endpoints return nothing for it.

Each run of the job (hourly) stores the position it saw (size and average entry). When the next run sees more, the
difference is a trade: its size is the change and its price follows from the average entry, since
entry_after * size_after = entry_before * size_before + price * added. When it sees less, the oldest trades
close first. A trade is known to fall between the two runs: ``after`` (the earlier run) and ``seen_at``.
Its signals are read from the close before ``after``'s day: the close a trade in that window saw, except when
the window spans 00:00 UTC (then it is the day before's, never a later one). A position already held at the
first run is one trade, ``first_seen``, whose real start is unknown.

The stored document holds sizes and prices, so it lives only in the dashboard's private state, never in a
log or this repository.
"""

from __future__ import annotations

import json

SCHEMA = 1
MAX_TRADES = 500
EPS = 1e-6


def load(text: str | None) -> dict:
    try:
        state = json.loads(text) if text else None
    except ValueError:
        state = None
    if not isinstance(state, dict) or state.get("v") != SCHEMA:
        return {"v": SCHEMA, "positions": {}, "trades": []}
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    return state


def dump(state: dict) -> str:
    state["trades"] = state["trades"][-MAX_TRADES:]
    return json.dumps(state, separators=(",", ":"))


def _close_oldest(trades: list[dict], product: str, amount: float) -> None:
    """Take ``amount`` (same sign as the position) off the product's oldest open trades."""
    for t in trades:
        if abs(amount) <= EPS:
            return
        if t["product"] != product or abs(t["size"]) <= EPS:
            continue
        take = amount if abs(amount) < abs(t["size"]) else t["size"]
        t["size"] -= take
        amount -= take


def update(state: dict, positions: list[dict], now: str) -> dict:
    """Fold one run's positions (product, size signed, entry) into the state; ``now`` is this run's time."""
    trades = state["trades"]
    seen = {p["product"]: p for p in positions if p.get("product") and p.get("size") is not None}
    for product in list(state["positions"]) + [k for k in seen if k not in state["positions"]]:
        before = state["positions"].get(product)
        now_p = seen.get(product)
        size = now_p["size"] if now_p else 0.0
        entry = now_p.get("entry") if now_p else None
        prev = before["size"] if before else 0.0
        if before is not None and prev and (abs(size) <= EPS or (size > 0) != (prev > 0)):
            _close_oldest(trades, product, prev)  # closed, or flipped through zero
            prev = 0.0
        if abs(size) > abs(prev) + EPS and entry:
            added = size - prev
            if prev and before.get("entry"):
                price = (entry * size - before["entry"] * prev) / added
            else:
                price = entry
            trades.append({"product": product, "size": added, "opened": added, "price": price, "seen_at": now,
                           "after": before["seen_at"] if before else None, "first_seen": before is None})
        elif abs(size) < abs(prev) - EPS:
            _close_oldest(trades, product, prev - size)
        state["positions"][product] = {"size": size, "entry": entry, "seen_at": now}
    state["trades"] = [t for t in trades if abs(t["size"]) > EPS]
    return state


def open_trades(state: dict, product: str) -> list[dict]:
    """The product's trades still open, oldest first, with the time their signals are read from: the run before
    they were seen, or for a position already held at the first run, the run that first saw it."""
    return [{**t, "time": t["after"] or t["seen_at"]} for t in state["trades"] if t["product"] == product and abs(t["size"]) > EPS]
