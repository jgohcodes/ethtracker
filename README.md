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
| `snapshot` | Price, returns, SMAs, RSI, ATR, realised vol, pullback gauges (distance below the 30-day high / above the 30-day low, z-score), and how far ETH historically moved *against* a long or short within your holding period |
| `study` | Simulates a rule-based pullback strategy on daily candles, one position at a time, in R-multiples, and compares it with buy-and-hold. See [Testing a pullback rule](#testing-a-pullback-rule) |
| `size` | Sizes a trade from its stop, tests it against your rules, prints a leverage ladder, price scenarios and history. **Exit code 1 if blocked** |
| `open` / `close` | Record a position you took / closed. `open` refuses rule-breaking plans unless `--force` (which is written into the note) |
| `risk` | Marks open positions to market: PnL, liquidation distance, stop distance, funding accrued, time held, portfolio open risk and drawdown. **Exit code 1 if any flag is raised** |
| `watch` | Polls the price; fires once on level crossings (`--above`, `--below`) and on position flags (`STOP_HIT`, `LIQ_DANGER`, `TIME_STOP`, ...). `--once` for cron |
| `equity`, `history` | Show/set equity; closed trades with R-multiples |

```
ethtracker snapshot
ethtracker study --trigger 15 --stop-atr 2 --target-r 2 --leverage 5    # history of your rule
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

`risk` also flags `OPEN_RISK_LIMIT`, `DRAWDOWN_LIMIT` (equity 20% below its recorded peak), `TIME_WARN` at 80% of `max_holding_days` (14) and `TIME_STOP` at 100%.

## Testing a pullback rule

`study` encodes "buy a large pullback, sell the bounce" as a rule and replays it over history:

```
ethtracker study --side long --lookback 30 --trigger 15 --stop-atr 2 --target-r 2 --leverage 5 --days 3000
```

A long enters at the close of the first day price is `--trigger`% below its `--lookback`-day high (a short mirrors this off the low), then exits at the stop, the target, liquidation or the time stop (`max_holding_days`), and re-enters only after exiting. Stops can be a percentage (`--stop-pct`) or a multiple of the 14-day ATR at entry (`--stop-atr`), which widens them exactly when volatility is high, as it usually is after a large drop. Results are in R-multiples with fees and funding included, plus the compounded return at your risk-per-trade and buy-and-hold over the same window.

Limits you should hold it to:

- Events are rare. A handful of qualifying pullbacks per year means dozens of trades even over several years, which is too few to trust a win rate. The tool says so when n < 30.
- Do not sweep `--trigger`, `--stop` and `--target` to find a winner. At this sample size that fits noise.
- "Undersold" and "overbought" as you judge them by eye are not testable; the drawdown trigger and target are a proxy for them. If your real rule differs (RSI, a support level, a catalyst), say so and encode it.
- Daily bars: fills at stop and target are exact (no slippage or gap-through), entry is the day's close, and a day touching both stop and target counts as the stop.

## Assumptions and limits

- **Model**: linear USD-margined perpetual, isolated margin, one flat maintenance margin rate. Real venues use tiered rates, liquidation fees and insurance funds. Treat `liq price` as an estimate and compare it with your venue's own number before trading. Cross margin with spare collateral liquidates further away, so isolated is the conservative case.
- **Defaults to verify against your venue**: `maintenance_margin_rate` 0.5%, `taker_fee_rate` 0.05%/side, `funding_rate_8h` 0.01%. Funding floats with the market and can turn negative; the estimate over `max_holding_days` is a rough scale. The shipped `max_leverage = 15` is not a Coinbase-specific figure: on Coinbase, which product you trade sets the leverage cap, contract size, fee tier, funding schedule and maintenance margin, so check that product's current terms and set `config.toml` to match.
- **Reward:risk** is net of fees and of funding over the full hold, which is conservative if the target is hit early.
- **Prices**: spot comes from the [Coinbase Prices API](https://docs.cdp.coinbase.com/coinbase-app/track-apis/prices) (`GET api.coinbase.com/v2/prices/ETH-USD/spot`, public, no key). That endpoint returns a single price, not highs and lows, so ATR, the adverse-move history and `study` use daily candles from Coinbase Exchange's public candles endpoint instead. Kraken is an alternative provider. All of this is spot: a perpetual's mark price can differ, and liquidation uses mark.
- **Historical stats** use daily candles and overlapping windows. They indicate scale, not probability. The Coinbase adapter pages through 300-candle requests until it has the days you ask for; Kraken's endpoint returns at most 720 candles.
- **Equity** is your wallet balance excluding open PnL. `close` updates it with estimated fees and funding unless you pass the actual `--fees` / `--funding`. Your exchange statement is authoritative; this journal is a discipline tool.
- **Not verified live**: the Coinbase and Kraken adapters follow their documented response formats and are tested against fixtures, but were built in an environment with no access to those hosts. The spot parser reads only `data.amount`. Run `ethtracker price` and `ethtracker snapshot` first; if either fails, the error names the URL.

## Toward a bot

`risk.py` is pure (no I/O) on purpose: `plan_position` is the gate a bot would call before every order, and `mark_position` / `portfolio_status` are its kill-switch checks. What is missing, in order:

1. **Backtester**: `study` is the first cut (same sizing, liquidation, fee and funding functions, daily bars, one position). A proper one adds intraday candles for stop/target ordering, slippage, walk-forward validation and a rule set you have committed to before looking at results. Nothing should trade live that has not survived this.
2. **Strategy interface**: `(candles, state) -> intent`, with the risk gate between intent and order.
3. **Paper broker**, run for weeks against live prices.
4. **Exchange adapter**, only after 1 to 3. Use API keys with trading but no withdrawal permission, keep them outside the repo, add a daily-loss kill-switch and a hard position cap enforced on the exchange side as well.

## Development

```
pytest
```
