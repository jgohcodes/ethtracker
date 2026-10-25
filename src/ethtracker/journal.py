"""Account state and position journal, persisted as JSON in the ethtracker home directory."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .risk import Position


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    def __init__(self, path: Path):
        self.path = path
        self.equity: float | None = None
        self.peak_equity: float | None = None
        self.next_id = 1
        self.positions: list[Position] = []
        self.plan: dict | None = None  # {"floor": USD, "weekly_add": USD}: the adding plan, if one is set
        self.decisions: list[dict] = []  # per buy: what the signals said when it was made (see decisions.py)
        if path.exists():
            raw = json.loads(path.read_text())
            self.equity = raw["equity"]
            self.peak_equity = raw["peak_equity"]
            self.next_id = raw["next_id"]
            self.positions = [Position(**p) for p in raw["positions"]]
            self.plan = raw.get("plan")
            self.decisions = raw.get("decisions", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "next_id": self.next_id,
            "positions": [asdict(p) for p in self.positions],
        }
        if self.plan:
            payload["plan"] = self.plan
        if self.decisions:
            payload["decisions"] = self.decisions
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.is_open]

    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions if not p.is_open]

    def get(self, position_id: int) -> Position:
        for p in self.positions:
            if p.id == position_id:
                return p
        raise KeyError(f"no position with id {position_id}")

    def set_equity(self, equity: float) -> None:
        """Deposits and withdrawals reset the baseline, so the drawdown peak resets with it."""
        self.equity = self.peak_equity = equity

    def deposit(self, amount: float) -> None:
        """New cash: it raises the wallet and the drawdown peak alike, so it never counts as a gain."""
        self.equity = (self.equity or 0.0) + amount
        self.peak_equity = (self.peak_equity or 0.0) + amount

    def set_plan(self, floor: float, weekly_add: float) -> None:
        if floor <= 0 or weekly_add < 0:
            raise ValueError("the floor must be above zero and the weekly add zero or more")
        self.plan = {"floor": float(floor), "weekly_add": float(weekly_add), "set_at": now_iso()}

    def open(self, opened_at: str | None = None, **fields) -> Position:
        pos = Position(id=self.next_id, opened_at=opened_at or now_iso(), **fields)
        self.next_id += 1
        self.positions.append(pos)
        return pos

    def record_decision(self, position_id: int, record: dict) -> None:
        """Keep one record per position: a later capture (a backfill) replaces an earlier one that failed."""
        self.decisions = [d for d in self.decisions if d.get("position_id") != position_id]
        self.decisions.append({"position_id": position_id, **record})

    def close(self, position_id: int, exit_price: float, realized_pnl: float) -> Position:
        pos = self.get(position_id)
        if not pos.is_open:
            raise ValueError(f"position {position_id} is already closed")
        pos.closed_at = now_iso()
        pos.exit_price = exit_price
        pos.realized_pnl = realized_pnl
        if self.equity is not None:
            self.equity += realized_pnl
            self.peak_equity = max(self.peak_equity or self.equity, self.equity)
        return pos
