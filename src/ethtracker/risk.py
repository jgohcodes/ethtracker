"""Position sizing, liquidation and portfolio risk maths for linear ETH-USD perpetuals.

Pure functions only, no I/O, so the same code can gate a bot later.

Model: isolated margin, linear contract (USD PnL = qty * price move), maintenance
margin charged on mark notional at one fixed rate. Venues differ (tiered
maintenance rates, liquidation fees, insurance funds), so ``liq_price`` is an
estimate. Confirm it against your venue's own liquidation price before trading.
Cross margin with spare collateral liquidates further away, so isolated is the
conservative choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import Config

FUNDING_PERIODS_PER_DAY = 3
FUNDING_WARN_R = 0.25  # warn when funding over the hold exceeds this multiple of the risk budget
DEFAULT_MOVES = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30)
_EPS = 1e-9


def _sign(side: str) -> int:
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    return 1 if side == "long" else -1


def pnl(side: str, entry: float, price: float, qty: float) -> float:
    return _sign(side) * (price - entry) * qty


def liquidation_price(side: str, entry: float, leverage: float, mmr: float) -> float:
    """Price at which equity (margin + PnL) falls to maintenance margin.

    long:  M + q(p - e) = m*q*p  ->  p = e(1 - 1/L) / (1 - m)
    short: M + q(e - p) = m*q*p  ->  p = e(1 + 1/L) / (1 + m)
    with M = q*e/L.
    """
    if leverage < 1:
        raise ValueError("leverage must be >= 1")
    if _sign(side) == 1:
        return entry * (1 - 1 / leverage) / (1 - mmr)
    return entry * (1 + 1 / leverage) / (1 + mmr)


def max_leverage_for_distance(side: str, stop_dist: float, mmr: float, fraction: float) -> float:
    """Highest leverage whose liquidation distance is at least ``stop_dist / fraction`` (fractions of entry)."""
    wanted = stop_dist / fraction
    if _sign(side) == 1:
        inverse = 1 - (1 - wanted) * (1 - mmr)
    else:
        inverse = (1 + wanted) * (1 + mmr) - 1
    return 1 / inverse


def max_leverage_for_stop(side: str, entry: float, stop: float, mmr: float, fraction: float) -> float:
    _check_levels(side, entry, stop, None)
    return max_leverage_for_distance(side, abs(entry - stop) / entry, mmr, fraction)


def funding_cost(side: str, notional: float, rate_8h: float, days: float) -> float:
    """Signed USD cost of funding: positive = you pay (longs pay when the rate is positive)."""
    return _sign(side) * notional * rate_8h * FUNDING_PERIODS_PER_DAY * days


def _check_levels(side: str, entry: float, stop: float, target: float | None) -> None:
    s = _sign(side)
    if entry <= 0 or stop <= 0:
        raise ValueError("entry and stop must be positive")
    if s * (entry - stop) <= 0:
        want = "below" if s == 1 else "above"
        raise ValueError(f"stop {stop} must be {want} entry {entry} for a {side}")
    if target is not None and s * (target - entry) <= 0:
        want = "above" if s == 1 else "below"
        raise ValueError(f"target {target} must be {want} entry {entry} for a {side}")


@dataclass(frozen=True)
class SizePlan:
    side: str
    entry: float
    stop: float
    target: float | None
    leverage: float
    qty: float
    notional: float
    margin: float
    margin_pct_equity: float
    effective_leverage: float  # notional / equity: the exposure that actually matters
    liq_price: float
    stop_dist: float  # fractions of entry
    liq_dist: float
    stop_fraction_of_liq: float
    max_leverage_for_stop: float
    stop_atr_multiple: float | None  # stop distance / 14d ATR, when ATR is known
    risk_usd: float  # loss if the stop fills, fees included
    risk_pct: float
    fees_at_stop_usd: float
    hold_days: float
    funding_usd: float  # signed estimate over hold_days, positive = cost
    reward_usd: float | None  # net of fees and funding
    reward_risk: float | None
    margin_capped: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def plan_position(
    cfg: Config,
    equity: float,
    side: str,
    entry: float,
    stop: float,
    leverage: float,
    *,
    target: float | None = None,
    risk_pct: float | None = None,
    qty: float | None = None,
    hold_days: float | None = None,
    open_risk_usd: float = 0.0,
    atr_pct: float | None = None,
) -> SizePlan:
    """Size a trade from its stop and risk budget, then test it against the rules in ``cfg``.

    With ``qty=None`` the size is derived from the risk budget (and capped by
    ``max_margin_pct``). With ``qty`` given, that size is evaluated as-is, which
    is how a fill you already have is checked. ``atr_pct`` is the 14d ATR as a
    fraction of price; without it the stop-vs-noise rule is skipped with a warning.
    """
    _check_levels(side, entry, stop, target)
    if equity <= 0:
        raise ValueError("equity must be positive")
    if leverage < 1:
        raise ValueError("leverage must be >= 1")

    fee = cfg.taker_fee_rate
    hold_days = cfg.max_holding_days if hold_days is None else hold_days
    budget_pct = cfg.risk_per_trade_pct if risk_pct is None else risk_pct
    margin_cap_usd = equity * cfg.max_margin_pct / 100

    margin_capped = False
    if qty is None:
        per_unit_loss = abs(entry - stop) + fee * (entry + stop)
        qty = equity * budget_pct / 100 / per_unit_loss
        qty_cap = margin_cap_usd * leverage / entry
        if qty > qty_cap:
            qty, margin_capped = qty_cap, True

    notional = qty * entry
    margin = notional / leverage
    liq = liquidation_price(side, entry, leverage, cfg.maintenance_margin_rate)
    stop_dist = abs(entry - stop) / entry
    liq_dist = abs(entry - liq) / entry
    fees_at_stop = fee * qty * (entry + stop)
    risk_usd = qty * abs(entry - stop) + fees_at_stop
    funding = funding_cost(side, notional, cfg.funding_rate_8h, hold_days)
    max_lev = max_leverage_for_stop(
        side, entry, stop, cfg.maintenance_margin_rate, cfg.max_stop_to_liq_fraction
    )

    reward_usd = reward_risk = None
    if target is not None:
        reward_usd = qty * abs(target - entry) - fee * qty * (entry + target) - funding
        reward_risk = reward_usd / risk_usd

    stop_fraction = stop_dist / liq_dist
    stop_atr = None if atr_pct is None else stop_dist / atr_pct
    violations: list[str] = []
    warnings: list[str] = []

    if cfg.min_stop_atr_multiple > 0:
        if stop_atr is None:
            warnings.append("ATR not available: stop-vs-daily-noise check skipped.")
        elif stop_atr < cfg.min_stop_atr_multiple * (1 - _EPS):
            min_dist = cfg.min_stop_atr_multiple * atr_pct
            lev_at_min = max_leverage_for_distance(
                side, min_dist, cfg.maintenance_margin_rate, cfg.max_stop_to_liq_fraction
            )
            violations.append(
                f"Stop is {stop_dist:.1%} from entry = {stop_atr:.1f}x the 14d ATR ({atr_pct:.1%}), "
                f"inside normal daily noise (minimum {cfg.min_stop_atr_multiple:g}x = {min_dist:.1%}). "
                f"A stop that wide supports at most {lev_at_min:.1f}x leverage."
            )

    if leverage > cfg.max_leverage + _EPS:
        violations.append(f"Leverage {leverage:g}x exceeds the {cfg.max_leverage:g}x cap.")
    if stop_fraction > cfg.max_stop_to_liq_fraction + _EPS:
        if stop_fraction >= 1:
            lead = f"Liquidation ({liq_dist:.1%} away) triggers BEFORE your stop ({stop_dist:.1%} away)."
        else:
            lead = (
                f"Stop uses {stop_fraction:.0%} of the distance to liquidation "
                f"(limit {cfg.max_stop_to_liq_fraction:.0%}): stop {stop_dist:.1%}, liquidation {liq_dist:.1%}."
            )
        violations.append(f"{lead} Max leverage for this stop: {max_lev:.1f}x.")
    if margin > margin_cap_usd * (1 + _EPS):
        violations.append(
            f"Margin is {margin / equity:.0%} of equity (limit {cfg.max_margin_pct:g}%)."
        )
    if risk_usd > equity * budget_pct / 100 * (1 + _EPS):
        violations.append(
            f"Loss at stop is {risk_usd / equity:.2%} of equity (budget {budget_pct:g}%)."
        )
    open_cap = equity * cfg.max_open_risk_pct / 100
    if open_risk_usd + risk_usd > open_cap * (1 + _EPS):
        violations.append(
            f"Total open risk would be {(open_risk_usd + risk_usd) / equity:.2%} of equity "
            f"(limit {cfg.max_open_risk_pct:g}%)."
        )

    if margin_capped:
        warnings.append(
            f"Size capped by the {cfg.max_margin_pct:g}% margin limit; risk is "
            f"{risk_usd / equity:.2%}, below the {budget_pct:g}% budget."
        )
    if funding > FUNDING_WARN_R * risk_usd:
        warnings.append(
            f"Funding over {hold_days:g}d is ~${funding:,.0f} = {funding / risk_usd:.2f}R "
            f"at {cfg.funding_rate_8h:.4%}/8h."
        )
    if reward_risk is not None and reward_risk < cfg.min_reward_risk:
        warnings.append(
            f"Net reward:risk is {reward_risk:.2f}, below the {cfg.min_reward_risk:g} minimum."
        )
    if target is None:
        warnings.append("No target given: reward:risk not evaluated.")

    return SizePlan(
        side=side, entry=entry, stop=stop, target=target, leverage=leverage,
        qty=qty, notional=notional, margin=margin, margin_pct_equity=margin / equity,
        effective_leverage=notional / equity, liq_price=liq, stop_dist=stop_dist,
        liq_dist=liq_dist, stop_fraction_of_liq=stop_fraction, max_leverage_for_stop=max_lev,
        stop_atr_multiple=stop_atr, risk_usd=risk_usd, risk_pct=risk_usd / equity, fees_at_stop_usd=fees_at_stop,
        hold_days=hold_days, funding_usd=funding, reward_usd=reward_usd,
        reward_risk=reward_risk, margin_capped=margin_capped,
        violations=tuple(violations), warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class Scenario:
    move: float
    price: float
    pnl: float
    pnl_pct_equity: float
    liquidated: bool


def scenarios(plan: SizePlan, equity: float, moves: tuple[float, ...] = DEFAULT_MOVES) -> list[Scenario]:
    """PnL if price moves by each fraction from entry. Liquidation loses the whole margin."""
    s = _sign(plan.side)
    out = []
    for move in moves:
        price = plan.entry * (1 + move)
        liquidated = s * (price - plan.liq_price) <= 0
        value = -plan.margin if liquidated else s * (price - plan.entry) * plan.qty
        out.append(Scenario(move, price, value, value / equity, liquidated))
    return out


@dataclass
class Position:
    id: int
    side: str
    entry: float
    qty: float
    leverage: float
    stop: float
    target: float | None
    opened_at: str
    mmr: float
    initial_risk_usd: float
    note: str = ""
    closed_at: str | None = None
    exit_price: float | None = None
    realized_pnl: float | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None

    @property
    def margin(self) -> float:
        return self.qty * self.entry / self.leverage


@dataclass(frozen=True)
class PositionStatus:
    position: Position
    price: float
    unrealized: float
    liq_price: float
    dist_to_liq: float  # fraction of current price; <= 0 means liquidated
    dist_to_stop: float  # fraction of current price; <= 0 means stop breached
    days_held: float
    funding_accrued: float  # estimate, positive = cost
    loss_to_stop_usd: float  # further loss from the current price to the stop, incl. exit fee
    flags: tuple[str, ...]


def mark_position(cfg: Config, pos: Position, price: float, now: datetime) -> PositionStatus:
    s = _sign(pos.side)
    liq = liquidation_price(pos.side, pos.entry, pos.leverage, pos.mmr)
    dist_to_liq = s * (price - liq) / price
    dist_to_stop = s * (price - pos.stop) / price
    days_held = (now - datetime.fromisoformat(pos.opened_at)).total_seconds() / 86400
    funding = funding_cost(pos.side, pos.qty * pos.entry, cfg.funding_rate_8h, days_held)
    loss_to_stop = max(0.0, s * (price - pos.stop) * pos.qty) + cfg.taker_fee_rate * pos.qty * pos.stop

    flags = []
    if dist_to_liq <= 0:
        flags.append("LIQUIDATED")
    elif dist_to_liq * 100 < cfg.liq_warn_pct:
        flags.append("LIQ_DANGER")
    if dist_to_stop <= 0:
        flags.append("STOP_HIT")
    if days_held >= cfg.max_holding_days:
        flags.append("TIME_STOP")
    elif days_held >= 0.8 * cfg.max_holding_days:
        flags.append("TIME_WARN")

    return PositionStatus(
        position=pos, price=price, unrealized=pnl(pos.side, pos.entry, price, pos.qty),
        liq_price=liq, dist_to_liq=dist_to_liq, dist_to_stop=dist_to_stop,
        days_held=days_held, funding_accrued=funding, loss_to_stop_usd=loss_to_stop,
        flags=tuple(flags),
    )


@dataclass(frozen=True)
class PortfolioStatus:
    equity: float  # wallet balance excluding unrealized PnL
    unrealized: float
    total_equity: float
    margin_used: float
    margin_utilisation: float  # margin / total equity
    gross_notional: float
    net_notional: float  # long positive, short negative
    effective_leverage: float  # gross notional / total equity
    open_risk_usd: float
    open_risk_pct: float
    drawdown: float  # from recorded peak, fraction
    flags: tuple[str, ...]


def portfolio_status(
    cfg: Config, equity: float, peak_equity: float | None, statuses: list[PositionStatus]
) -> PortfolioStatus:
    unrealized = sum(s.unrealized for s in statuses)
    total = equity + unrealized
    margin = sum(s.position.margin for s in statuses)
    gross = sum(s.position.qty * s.price for s in statuses)
    net = sum(_sign(s.position.side) * s.position.qty * s.price for s in statuses)
    open_risk = sum(s.loss_to_stop_usd for s in statuses)
    peak = max(peak_equity or total, total)
    drawdown = 1 - total / peak if peak > 0 else 0.0

    flags = []
    if total > 0 and open_risk / total * 100 > cfg.max_open_risk_pct + _EPS:
        flags.append("OPEN_RISK_LIMIT")
    if drawdown * 100 >= cfg.max_account_drawdown_pct:
        flags.append("DRAWDOWN_LIMIT")

    return PortfolioStatus(
        equity=equity, unrealized=unrealized, total_equity=total, margin_used=margin,
        margin_utilisation=margin / total if total > 0 else float("inf"),
        gross_notional=gross, net_notional=net,
        effective_leverage=gross / total if total > 0 else float("inf"),
        open_risk_usd=open_risk, open_risk_pct=open_risk / total if total > 0 else float("inf"),
        drawdown=drawdown, flags=tuple(flags),
    )
