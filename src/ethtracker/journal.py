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
        if path.exists():
            raw = json.loads(path.read_text())
            self.equity = raw["equity"]
            self.peak_equity = raw["peak_equity"]
            self.next_id = raw["next_id"]
            self.positions = [Position(**p) for p in raw["positions"]]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "next_id": self.next_id,
            "positions": [asdict(p) for p in self.positions],
        }
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

    def open(self, **fields) -> Position:
        pos = Position(id=self.next_id, opened_at=now_iso(), **fields)
        self.next_id += 1
        self.positions.append(pos)
        return pos

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
