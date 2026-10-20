"""Static risk parameters, read from ``config.toml`` in the ethtracker home directory.

Home is ``$ETHTRACKER_HOME`` or ``~/.ethtracker``. Fields ending in ``_pct`` are
percent units (1.0 = 1%); ``*_rate`` and ``*_fraction`` fields are plain fractions.
Mutable account state (equity, positions) lives in ``state.json``, not here.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("ETHTRACKER_HOME", Path.home() / ".ethtracker"))


@dataclass(frozen=True)
class Config:
    risk_per_trade_pct: float = 1.0
    max_open_risk_pct: float = 3.0
    max_leverage: float = 15.0
    max_stop_to_liq_fraction: float = 0.5
    min_stop_atr_multiple: float = 2.0
    max_margin_pct: float = 50.0
    min_reward_risk: float = 2.0
    max_holding_days: int = 90
    max_account_drawdown_pct: float = 20.0
    liq_warn_pct: float = 3.0
    maintenance_margin_rate: float = 0.005
    taker_fee_rate: float = 0.0005
    funding_rate_8h: float = 0.0001
    provider: str = "coinbase"


COMMENTS = {
    "risk_per_trade_pct": "Max loss if the stop is hit, % of equity (fees included)",
    "max_open_risk_pct": "Cap on summed loss-to-stop across all open positions, % of equity",
    "max_leverage": "Hard cap on the exchange leverage setting",
    "max_stop_to_liq_fraction": "Stop must sit within this fraction of the distance to liquidation",
    "min_stop_atr_multiple": "Stop must be at least this many 14d ATRs from entry (0 disables): tighter sits inside daily noise",
    "max_margin_pct": "Max margin posted to a single position, % of equity",
    "min_reward_risk": "Minimum net reward:risk (after fees and funding) when a target is given",
    "max_holding_days": "Time stop: your 3-month swing horizon",
    "max_account_drawdown_pct": "Flag when equity is this far below its recorded peak",
    "liq_warn_pct": "Flag when price is within this many % of liquidation",
    "maintenance_margin_rate": "Verify against your venue's tier table; varies by notional",
    "taker_fee_rate": "Per side, as a fraction (0.0005 = 0.05%). Verify against your fee tier",
    "funding_rate_8h": "Assumed funding per 8h paid by longs (0.0001 = 0.01%). Floats with the market",
    "provider": "Price source: coinbase | kraken | csv",
}


def load(path: Path | None = None) -> Config:
    path = path or home() / "config.toml"
    if not path.exists():
        return Config()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(unknown))}")
    return Config(**raw)


def render_default_toml() -> str:
    defaults = Config()
    lines = []
    for f in fields(Config):
        value = getattr(defaults, f.name)
        rendered = f'"{value}"' if isinstance(value, str) else repr(value)
        lines.append(f"# {COMMENTS[f.name]}")
        lines.append(f"{f.name} = {rendered}")
        lines.append("")
    return "\n".join(lines)
