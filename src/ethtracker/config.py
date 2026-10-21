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
    stress_drop_pct: float = 40.0
    max_stress_loss_pct: float = 100.0
    min_reward_risk: float = 2.0
    max_holding_days: int = 14
    max_account_drawdown_pct: float = 20.0
    liq_warn_pct: float = 3.0
    maintenance_margin_rate: float = 0.005
    taker_fee_rate: float = 0.0005
    funding_rate_8h: float = 0.0001
    margin_mode: str = "cross"
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    extension_z: float = 2.0
    vol_spike_ratio: float = 1.5
    range_spike_atr: float = 2.0
    oi_spike_pct: float = 10.0
    oi_high_percentile: float = 90.0
    funding_hot_8h: float = 0.0005
    liq_spike_z: float = 2.0
    deep_itm_pct: float = 10.0
    exit_risk_flags: int = 2
    trail_atr_multiple: float = 3.0
    derivs_instrument: str = "ETH_USDC-PERPETUAL"
    provider: str = "coinbase"

    def __post_init__(self):
        if self.margin_mode not in ("cross", "isolated"):
            raise ValueError(f"margin_mode must be 'cross' or 'isolated', got {self.margin_mode!r}")


COMMENTS = {
    "margin_mode": "cross: one balance backs every position; isolated: each position has its own margin. Confirm on your account",
    "rsi_overbought": "RSI at or above this is a bearish signal for longs",
    "rsi_oversold": "RSI at or below this is a bullish signal for longs",
    "extension_z": "Std devs from the 20d mean at which price counts as stretched (a risk flag)",
    "vol_spike_ratio": "Volatility counts as spiking when 7d/30d realised vol, or ATR/its 180d median, reaches this ratio (a risk flag)",
    "range_spike_atr": "Also a volatility spike when the latest candle's range is this many prior ATRs",
    "oi_spike_pct": "A 24h open-interest change of this size (%) is a spike",
    "oi_high_percentile": "Open interest at or above this percentile of its recorded history is crowded (a risk flag; needs 30 days)",
    "funding_hot_8h": "Funding per 8h beyond +/- this is crowded (0.0005 = 0.05%). A folk threshold, not a validated one",
    "liq_spike_z": "A day's liquidations this many std devs above the trailing mean is a spike (needs a liquidation feed)",
    "deep_itm_pct": "A position this far (%) in profit from entry is 'deep in the money' and gets profit-protection rules",
    "exit_risk_flags": "Deep in the money plus this many risk flags recommends EXIT",
    "trail_atr_multiple": "Suggested trailing stop sits this many ATRs from the best price since entry",
    "derivs_instrument": "Deribit instrument whose open interest and funding the recorder polls (the linear USDC perpetual)",
    "risk_per_trade_pct": "Max loss if the stop is hit, % of equity (fees included)",
    "max_open_risk_pct": "Cap on summed loss-to-stop across all open positions, % of equity",
    "max_leverage": "Hard cap on the exchange leverage setting",
    "max_stop_to_liq_fraction": "Isolated margin only: stop must sit within this fraction of the distance to liquidation",
    "min_stop_atr_multiple": "Stop must be at least this many 14d ATRs from entry (0 disables): tighter sits inside daily noise",
    "max_margin_pct": "Max margin posted to a single position, % of equity",
    "stress_drop_pct": "The book must survive a move this big against it: whole book if cross, each position's leverage if isolated (0 disables)",
    "max_stress_loss_pct": "Also cap the equity lost at that move, % of equity; 100 = only the margin-limit test",
    "min_reward_risk": "Minimum net reward:risk (after fees and funding) when a target is given",
    "max_holding_days": "Time stop and funding/history horizon: how long you hold a swing, in days",
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
