# ethtracker

ETH-USD price tracker, leveraged position sizer and risk journal. A trading bot is the planned next layer; see [Toward a bot](#toward-a-bot).

```
pip install -e '.[dev]'        # Python 3.11+
ethtracker init --equity 25000 # writes ~/.ethtracker/config.toml, records account equity
ethtracker price               # first run: confirms your network can reach the price source
```

State lives in `~/.ethtracker` (override with `ETHTRACKER_HOME`): `config.toml` holds the rules, `state.json` holds equity and positions.

## Commands

| Command | What it does |
|---|---|
| `snapshot` | Price, returns, SMAs, RSI, ATR, realised vol, and how far ETH historically moved *against* a 90-day long or short from entry |
| `size` | Sizes a trade from its stop, tests it against your rules, prints a leverage ladder, price scenarios and history. **Exit code 1 if blocked** |
| `open` / `close` | Record a position you took / closed. `open` refuses rule-breaking plans unless `--force` (which is written into the note) |
| `risk` | Marks open positions to market: PnL, liquidation distance, stop distance, funding accrued, time held, portfolio open risk and drawdown. **Exit code 1 if any flag is raised** |
| `watch` | Polls the price; fires once on level crossings (`--above`, `--below`) and on position flags (`STOP_HIT`, `LIQ_DANGER`, `TIME_STOP`, ...). `--once` for cron |
| `equity`, `history` | Show/set equity; closed trades with R-multiples |

```
ethtracker snapshot
ethtracker size --side long --stop 2850 --target 3600 --leverage 5      # entry defaults to live spot
ethtracker open --side long --entry 3012 --stop 2850 --leverage 5 --qty 1.2
ethtracker risk
ethtracker watch --above 3500 --below 2900
```

Offline or with your own data: `--provider csv --csv eth.csv` (columns `date|time`, `open`, `high`, `low`, `close`).

## How sizing works

Size comes from the **stop and the risk budget**, not from leverage:

```
qty = equity * risk% / ( |entry - stop| + fee * (entry + stop) )
```

Leverage only sets how much margin you post and how far away liquidation sits. For an isolated linear perpetual with maintenance margin rate `m`:

```
long  liq = entry * (1 - 1/L) / (1 - m)        short liq = entry * (1 + 1/L) / (1 + m)
```

At 10x the liquidation is about 9.5% from entry, at 15x about 6.2% (with `m` = 0.5%). A trade is **blocked** if any of these fail (all configurable in `config.toml`):

1. Leverage above `max_leverage` (15).
2. Stop uses more than `max_stop_to_liq_fraction` (50%) of the distance to liquidation.
3. Stop is closer than `min_stop_atr_multiple` (2) times the 14-day ATR, i.e. inside normal daily noise. Set to 0 to disable; skipped with a warning if candles can't be fetched.
4. Loss at stop exceeds the risk budget (`risk_per_trade_pct`, 1%), margin exceeds `max_margin_pct` (50%), or summed loss-to-stop across open positions exceeds `max_open_risk_pct` (3%).

Rules 2 and 3 together are the point: a stop wide enough to survive ETH's noise needs low leverage, and the leverage ladder in `size` shows where that boundary is for your actual stop. The defaults are my starting values, not recommendations; change them deliberately.

`risk` also flags `OPEN_RISK_LIMIT`, `DRAWDOWN_LIMIT` (equity 20% below its recorded peak), `TIME_WARN` at 80% of `max_holding_days` and `TIME_STOP` at 100%.

## Assumptions and limits

- **Model**: linear USD-margined perpetual, isolated margin, one flat maintenance margin rate. Real venues use tiered rates, liquidation fees and insurance funds. Treat `liq price` as an estimate and compare it with your venue's own number before trading. Cross margin with spare collateral liquidates further away, so isolated is the conservative case.
- **Defaults to verify against your venue**: `maintenance_margin_rate` 0.5%, `taker_fee_rate` 0.05%/side, `funding_rate_8h` 0.01%. Funding floats with the market and can turn negative; the 90-day estimate is a rough scale.
- **Reward:risk** is net of fees and of funding over the full hold, which is conservative if the target is hit early.
- **Prices** are Coinbase ETH-USD spot (Kraken as fallback). A perpetual's mark price can differ from spot, and liquidation uses mark.
- **Historical stats** use daily candles and overlapping windows. They indicate scale, not probability. The Coinbase adapter pages through 300-candle requests until it has the days you ask for; Kraken's endpoint returns at most 720 candles.
- **Equity** is your wallet balance excluding open PnL. `close` updates it with estimated fees and funding unless you pass the actual `--fees` / `--funding`. Your exchange statement is authoritative; this journal is a discipline tool.
- **Not verified live**: the Coinbase and Kraken adapters follow their documented response formats and are tested against fixtures, but were built in an environment with no access to those hosts. Run `ethtracker price` and `ethtracker snapshot` first; if either fails, the error names the URL.

## Toward a bot

`risk.py` is pure (no I/O) on purpose: `plan_position` is the gate a bot would call before every order, and `mark_position` / `portfolio_status` are its kill-switch checks. What is missing, in order:

1. **Backtester** over the same candles, using the same sizing and liquidation functions, including funding and fees. Nothing should trade live that has not survived this.
2. **Strategy interface**: `(candles, state) -> intent`, with the risk gate between intent and order.
3. **Paper broker**, run for weeks against live prices.
4. **Exchange adapter**, only after 1 to 3. Use API keys with trading but no withdrawal permission, keep them outside the repo, add a daily-loss kill-switch and a hard position cap enforced on the exchange side as well.

## Development

```
pytest
```
