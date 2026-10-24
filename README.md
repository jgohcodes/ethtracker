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
| `sync` / `export` | `sync` downloads daily candles into a local cache (later runs fetch only the gap). `export --out eth.csv` writes candles plus SMA 20/50/100/200, RSI, ATR, volatility and volume average for another app. See [Data](#data) |
| `headroom` | How much more you can add and still survive the stress move, from current equity and holdings. See [The stress rule](#the-stress-rule) |
| `size` | Sizes a trade from its stop and your remaining headroom, tests it against your rules, prints price scenarios and history (plus a leverage ladder under isolated margin). **Exit code 1 if blocked** |
| `open` / `close` | Record a position you took / closed. `open` refuses rule-breaking plans unless `--force` (which is written into the note) |
| `risk` | Marks open positions to market: PnL, liquidation distance, stop distance, funding accrued, time held, portfolio open risk and drawdown. **Exit code 1 if any flag is raised** |
| `signals` | Trend, RSI, volatility, anchored VWAP, volume-profile point of control, and (when recorded or supplied) open-interest, funding and liquidation positioning. See [Signals and recommended actions](#signals-and-recommended-actions) |
| `advise` | Per open position: a recommended action (HOLD, TIGHTEN, TAKE_PARTIAL, EXIT), the bull case, the bear case and why the rule fired. **Exit code 1 on EXIT or TAKE_PARTIAL** |
| `record` | Polls Deribit's public ticker and appends open interest and funding to a local history, which is what makes the open-interest signals possible |
| `signals-study` | Tests the candle-derived risk flags against history: what followed days with 0, 1, 2+ flags |
| `cycle-study` | Dip-buying with leverage versus holding spot: inside each bull phase of the 4-year cycle (found with hindsight) and over the whole history with a 200-day trend filter |
| `combo-study` | A long-only perp core (held or kept at its leverage) plus a small 10-15x swing sleeve whose profits are swept into it, against spot: by cycle phase, over 4-year windows from deep drawdowns, with funding sensitivity and the growth arithmetic of leverage |
| `short-study` | Short-term shorts when ETH looks overheated (the dashboard's 2+ flags, RSI 80+, funding 0.05%/8h+ from Deribit's funding history) at 1-15x, and the same signals used to cut the 2.5x core to 1x until it cools |
| `swing-study` | Buy a 20-25% pullback (or the first recovery after one), sell only into a spike (RSI 80+ or a 30% run in 14 days), on `--asset ETH` or `BTC`, against holding spot, with Deribit funding history |
| `trim-study` | Trim a third or half into spikes and refill on 20% dips, with the liquidation price pinned where a 2.5x or 1.5x position started, on `--asset ETH` or `BTC` |
| `pullback-study` | How often "the pullback is over" signals were right (the low held, price higher 30/90/365 days on, with a 95% lower bound), buy orders 5-20% under the price, and adding every week versus saving for the signal, on `--asset ETH` or `BTC` |
| `timing-study` | Buying every week but choosing the day within a 4-day window (live rules against the hindsight best and worst day), how often price broke a floor under the start within 1, 2 and 4 years, and with `--floor PRICE` the most leverage each new-cash add can carry without lifting the liquidation price |
| `publish` / `cloud-run` / `dashboard-json` | `publish` sends the dashboard snapshot from your machine and pushes your journal (run it after each trade); `cloud-run` is the daily job GitHub Actions runs; `dashboard-json` prints what would be sent. See [Dashboard](#dashboard) |
| `watch` | Polls the price; fires once on level crossings (`--above`, `--below`) and on position and whole-book flags. With open positions it recomputes each recommended action every poll and announces only when one changes. `--record` also logs derivatives data; `--once` for cron |
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

How leverage matters depends on `margin_mode` (default `cross`; confirm it on your account). Under **isolated** margin each position has its own margin and, for a linear perpetual with maintenance margin rate `m`:

```
long  liq = entry * (1 - 1/L) / (1 - m)        short liq = entry * (1 + 1/L) / (1 + m)
```

At 10x that is about 9.5% from entry, at 15x about 6.2% (with `m` = 0.5%). Under **cross** margin one balance backs every position, so a position's own isolated liquidation price is not what liquidates you: the whole book is, and the exchange leverage setting stops mattering (see [The stress rule](#the-stress-rule)). A trade is **blocked** if any of these fail (all configurable in `config.toml`):

1. Leverage above `max_leverage` (15).
2. Isolated margin only: stop uses more than `max_stop_to_liq_fraction` (50%) of the distance to liquidation.
3. Stop is closer than `min_stop_atr_multiple` (2) times the 14-day ATR, i.e. inside normal daily noise. Set to 0 to disable; skipped with a warning if candles can't be fetched.
4. Loss at stop exceeds the risk budget (`risk_per_trade_pct`, 1%), margin exceeds `max_margin_pct` (50%), or summed loss-to-stop across open positions exceeds `max_open_risk_pct` (3%).
5. The [stress rule](#the-stress-rule).

Under isolated margin, rules 2 and 3 together are the point: a stop wide enough to survive ETH's noise needs low leverage, and the leverage ladder in `size` shows where that boundary is for your actual stop. The defaults are my starting values, not recommendations; change them deliberately.

`risk` also flags `OPEN_RISK_LIMIT`, `DRAWDOWN_LIMIT` (equity 20% below its recorded peak), `TIME_WARN` at 80% of `max_holding_days` (14) and `TIME_STOP` at 100%. `LIQ_DANGER` and `LIQUIDATED` come from each position's own liquidation price under isolated margin, and from the whole book's liquidation price under cross margin.

## The stress rule

`stress_drop_pct` (default 40) asks: if price moves that far against the whole book, does equity still cover maintenance margin? Under cross margin, with total equity `E`, maintenance rate `m` and a long book of notional `N`, the book survives a fall `d` only if

```
N <= E / (d + m(1 - d))        # d = 40%, m = 0.5%:  N <= 2.48 x equity
```

`headroom` and `size` solve this for how much more you can add, from current equity and your open positions marked at the current price, so each entry is sized after the pullback, not before it. `size` caps a position at the smallest of the risk-budget size, the margin limit and this headroom.

Three things to understand before relying on it:

1. **Surviving the margin limit is not surviving.** At exactly 2.48x equity, a 40% fall costs 99% of equity (0.4 x 2.48). The tool prints this loss on every `size` and warns above 50%. Set `max_stress_loss_pct` to cap it (50 limits notional to 1.25x equity).
2. **It is not a standing guarantee.** Equity falls faster than notional, so a book sized to the limit is in breach after a smaller further fall. `risk` and `watch` raise `STRESS_BREACH` when that happens. With stop-based 1% sizing you will normally sit far below the limit; the rule binds when you size by deployment instead of by stop.
3. **Cross versus isolated.** The pooled maths needs cross margin. Under isolated margin spare balance does not protect a position, so the rule instead requires each position's own liquidation to lie beyond the move: leverage <= 1 / (1 - (1-d)(1-m)), which is 2.5x at d = 40%.

It ignores fees, funding, basis, tiered maintenance margin and venue liquidation fees, so real room is smaller.

## Signals and recommended actions

`signals` prints conditions at the latest candle. Each has a direction for price (`+`, `-`, `·`) and, where relevant, a risk flag (`!`) for conditions that raise the odds of a violent move in either direction:

| Family | Signals |
|---|---|
| levels | 20d vs 50d trend; price vs the 200d; anchored VWAP from the 365-day low (or `--anchor DATE`) and from the year open; the 365-day volume-profile point of control and value area |
| momentum | RSI overbought/oversold and a rollover from overbought; extension (std devs above the 20d mean) |
| volume | volume on down days vs up days over 10 days |
| volatility (risk flag only) | 7d/30d realised volatility, ATR against its 180-day median, or the latest range as a multiple of ATR |
| derivatives | funding per 8h; 24h open-interest change against price; open-interest percentile; long and short liquidation spikes |

Correlated signals share a family and vote once, so "below the 20d, 50d, 200d, both VWAPs and the POC" is one bearish vote, not six. Anchored VWAP and the volume profile are built from daily candles (each day's volume spread evenly over its range), so they are coarser than tick or intraday versions, and neither is on-chain data.

`advise` turns the signals into a recommended action per open position, with the **bull case** (what supports holding), the **bear case** (what argues against, including every risk flag) and the reasons the rule fired. `watch` recomputes it every poll and announces only when the action changes.

| Action | Fires on (first match wins) |
|---|---|
| EXIT | stop breached; time stop; book at or near liquidation; deep in the money with at least `exit_risk_flags` (2) risk flags |
| TAKE_PARTIAL (close 50%, trail the rest) | book outside the stress rule; deep in the money (at least `deep_itm_pct`, 10%) with any risk flag or a bearish net vote; in profit with 2+ risk flags and no net support |
| TIGHTEN | in profit with any risk flag or a bearish net vote; deep in the money with the bull case intact (trail only) |
| HOLD | otherwise, including an underwater position whose signals are deteriorating (honour the stop, do not add) |

The suggested stop is a chandelier: `trail_atr_multiple` (3) ATRs behind the best price since entry, or breakeven once a full ATR in profit. It never loosens a stop.

**"Automatic" means alert, not order.** Nothing here places an order. This decision engine is what a bot would call, but execution needs an exchange adapter that does not exist and should only follow paper trading.

What to trust:

- **Every threshold is a starting hypothesis** (`config.toml`). The derivatives signals are marked *(unvalidated)*: free history of open interest and liquidations is too short or absent to test them. "Funding above 0.05% per 8h is a high-accuracy short trigger" and "a long-liquidation spike marks the exact bottom" are folk rules, not tested here. Funding can stay elevated through a long trend, and a liquidation spike can be the first of several.
- **`signals-study` tests the candle-only flags** (stretched, RSI hot, volatility spike): for each flag count, how often a 15% or 20% drawdown followed within the horizon versus all days, and how often a 10% rally followed, which is the upside an early exit gives up. Read the `days` and `episodes` columns: flagged days are few and cluster (a 3-day flagged stretch is one episode, not three pieces of evidence), and small samples make noise look like signal. Run it on real history before believing that these flags "exit before the big candle":

  ```
  ethtracker sync --days 3000
  ethtracker signals-study --days 3000 | tee study.txt
  ```
- **Expect false exits.** A warning is at best a shift in odds, and each false exit costs upside.
- **Derivatives data is a snapshot, not a history.** Deribit's public ticker gives the current open interest and funding, so `record` builds the history: funding works immediately, the 24h open-interest change needs 24 hours of recording, and the percentile needs 30 days. There is no complete free liquidation feed (Binance's stream pushes at most one liquidation per symbol per second), so liquidation signals appear only if you pass `--derivs-csv` with `long_liq_usd` / `short_liq_usd` columns from a data vendor. `--derivs-csv` also accepts `open_interest`, `funding_8h` and the other columns from any source, with `time` or `date` as the first column.

## Dashboard

A private page on Vercel with your positions (each with its action, bull case and bear case), account and stress tiles, the signals, a price ladder and a TradingView daily chart. It runs on real data, refreshed once a day just after the 00:00 UTC daily close, which matches a 1D strategy: signals are computed on **completed daily candles only**, so an unfinished day never repaints them. Positions are marked at the spot price when the job runs.

```
GitHub Actions, daily at 00:17 UTC            Vercel (web/)                        your browser
  ethtracker cloud-run                          api/ingest  ──► Upstash Redis ◄──  api/snapshot ◄── page + TradingView
   ├ pulls your journal + OI history  ◄──────── api/state   ◄─┘   (snapshot, journal,     (DASHBOARD_TOKEN)
   ├ Coinbase daily candles, Deribit OI/funding                      OI/funding history)
   └ computes, publishes ──INGEST_TOKEN──────► api/ingest
your machine
  ethtracker open/close/equity, then `ethtracker publish --once` ──► snapshot + journal
```

Vercel only stores and displays; the maths runs in the job, which never holds an exchange key. Your journal and the open-interest history live in your Redis, never in this (public) repository.

**Set up (about ten minutes)**

1. **Vercel:** import this repository, set **Root Directory** to `web` (framework preset "Other"; `web/vercel.json` serves `public/`).
2. **Storage:** in the project, Storage (Marketplace), add **Upstash Redis** and connect it. It sets either `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` or `KV_REST_API_URL` / `KV_REST_API_TOKEN`; the API accepts either.
3. **Tokens:** add two environment variables with different random values (`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`): `DASHBOARD_TOKEN` (the browser reads with it) and `INGEST_TOKEN` (the job and your machine write with it). Redeploy so they take effect.
4. **GitHub:** repository Settings, Secrets and variables, Actions: add `ETHTRACKER_DASHBOARD_URL` (your `https://….vercel.app` address) and `ETHTRACKER_INGEST_TOKEN` (the same `INGEST_TOKEN`).
5. **First run:** Actions tab, "Daily dashboard", Run workflow. After that it runs every day at 00:17 UTC (GitHub may start scheduled jobs late, and pauses schedules in public repositories after 60 days without commits).
6. **Your positions:** on your machine, record trades as usual (`ethtracker equity`, `open`, `close`), then run

   ```
   export ETHTRACKER_DASHBOARD_URL=https://your-app.vercel.app
   export ETHTRACKER_INGEST_TOKEN=<the INGEST_TOKEN>
   ethtracker publish --once
   ```
   This updates the page immediately and pushes your journal, so every daily run marks the same positions. Run it after every trade.
7. Open the site and paste the `DASHBOARD_TOKEN`.

Without the GitHub secrets the job still runs as a **dry run**: live data, computed, nothing published. That is a quick check that the data sources work from GitHub's runners. "Run workflow" with *study* ticked also prints `signals-study`, backtests the pullback entry rule (20% and 15% below the 30-day high, 3x ATR stop, 14-day time stop, at 10x and 2.5x) and runs `cycle-study`, `combo-study`, `short-study`, `swing-study`, `trim-study`, `pullback-study` and `timing-study` (ETH and BTC) on the full daily history. The *studies* box picks some of them, e.g. `pullbacks,trim`.

**Code changes versus data.** The daily numbers reach the page through the API, so nothing is redeployed for them. Vercel redeploys only when the code changes. On the Hobby plan a commit whose author is not the Vercel account owner (commits made by an assistant included) can wait for your approval in Vercel before it deploys, and in a private repository it is refused; redeploy from the Vercel dashboard if that happens.

**Freshness.** The page judges age against the cadence the snapshot declares: a daily snapshot is "Updated" for 30 hours, "Late" after that and "Stale" after 54, with a banner saying where to look. The time is the server's receive time, which a client cannot forge.

**Derivatives history.** Deribit's public API gives only today's open interest and funding, so each daily run appends one row to the history kept in Redis: funding signals work from the first run, the 24h open-interest change from the second, and the open-interest percentile after 30 days. There is no free liquidation feed, so liquidation signals stay absent unless you supply one with `--derivs-csv` on your machine.

**Security.**

- The page is public; all data sits behind tokens. On Vercel's Hobby plan the production domain stays public (Vercel Authentication covers only previews there), so `DASHBOARD_TOKEN` is the lock. Treat it like a password; rotate by changing the variable and redeploying.
- The browser's token can read only the snapshot. It cannot write, and it cannot read the raw journal (`/api/state` takes the write token).
- The job's logs are public because the repository is: `cloud-run` prints market facts only (the close, signal counts, whether a journal was found), never equity, sizes or even how many positions are open, and GitHub masks the secrets.
- `publish` and `cloud-run` refuse plain `http://` (except localhost) and read tokens from the environment only. The page inserts everything as text, never markup. No Content-Security-Policy is set: a correct one needs testing in a browser against TradingView's domains.

**The chart** is TradingView's free embedded widget on the daily interval, defaulting to `COINBASE:ETHUSD` (the same spot data the signals use). Use `?symbol=EXCHANGE:SYMBOL` to change it. The widget cannot draw your entries, stops or the computed levels, so the price ladder lists them in price order beside the mark.

**What is and is not verified.** The API handlers are tested against a local HTTP server with an in-memory Redis (`cd web && npm install && npm test`); `publish`, `cloud-run` and the snapshot in the Python suite; the page in headless Chromium, end to end from the real commands. The job's data path is exercised on GitHub's runners by the workflow itself. The Vercel deployment, Upstash and the live TradingView widget are exercised only once you deploy.

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

## Data

Free, keyless sources:

| Need | Source | Notes |
|---|---|---|
| Spot price | Coinbase Prices API | a single price, no highs or lows |
| Daily open/high/low/close/volume | Coinbase Exchange public candles, ETH-USD | 300 candles per request, paged to the days you ask for |
| SMA 20/50/100/200, RSI, ATR, volatility, volume average | computed locally (`indicators.py`) | no free API serves these dependably; computing them from candles keeps them consistent, unmetered and auditable |
| Open interest and funding | Deribit public ticker, polled by `ethtracker record` | snapshots only; history accrues from the day you start |
| Alternatives | Kraken OHLC (last 720 days), your own CSV | `--provider kraken` / `--provider csv --csv FILE` |

Binance is not used. Its ETHUSDT market is USDT-quoted rather than USD, open interest and funding are venue-specific so they should come from the venue you trade, and its liquidation stream is a one-per-second snapshot. The "pull raw candles and compute indicators yourself" approach is what this tool already does; RSI and ATR match Wilder's published recursions (within 2e-5 after the seeding transient) and the SMAs match exactly.

`ethtracker sync` downloads once into `~/.ethtracker/candles/`. Every later command reads that cache and fetches only the gap (`--no-cache` bypasses it, and a failed refresh falls back to the cache with a warning). `ethtracker export --out eth.csv` writes the whole series with every metric for another app. Volume is in ETH, and the latest day is partial until it closes.

## Assumptions and limits

- **Venue**: Coinbase's help center reportedly says Coinbase International Exchange perpetuals migrated to a Deribit-powered venue on 1 October 2026, with API trading moving to a new gateway and new API keys ([Coinbase FAQ](https://help.coinbase.com/en/international-exchange/deribit/coinbase-faqs); I saw only search summaries of it and could not open the page). Do not build on the legacy International Exchange market-data endpoints. Which instrument you trade (linear USDC-margined or inverse coin-margined), its leverage cap, funding schedule and margin mode decide whether this tool's maths applies, so set `margin_mode`, `max_leverage`, `maintenance_margin_rate` and the fee from your product's terms. The ETHUSDC perpetual is linear and USDC-margined, which is the contract this tool's liquidation maths assumes. Deribit's name for it is `ETH_USDC-PERPETUAL` ([Deribit help center](https://support.deribit.com/hc/en-us/articles/31424969384605)), the recorder's default; Coinbase's developer docs describe a `{BASE}_USDC-PERPETUAL` pattern for migrated perpetuals, so confirm the exact name with Deribit's `public/get_instruments` (a wrong name makes `record` fail loudly).
- **Model**: a linear (USD- or USDC-margined) perpetual with one flat maintenance margin rate. Inverse (coin-margined) contracts are not modelled. `margin_mode` selects pooled (cross) or per-position (isolated) liquidation. Real venues use tiered rates, liquidation fees and insurance funds, so treat every liquidation figure as an estimate and compare it with your venue's own before trading.
- **Defaults to verify against your venue**: `maintenance_margin_rate` 0.5%, `taker_fee_rate` 0.05%/side, `funding_rate_8h` 0.01%. Funding floats with the market and can turn negative; the estimate over `max_holding_days` is a rough scale. The shipped `max_leverage = 15` is not a Coinbase-specific figure: on Coinbase, which product you trade sets the leverage cap, contract size, fee tier, funding schedule and maintenance margin, so check that product's current terms and set `config.toml` to match.
- **Reward:risk** is net of fees and of funding over the full hold, which is conservative if the target is hit early.
- **Prices**: spot comes from the [Coinbase Prices API](https://docs.cdp.coinbase.com/coinbase-app/track-apis/prices) (`GET api.coinbase.com/v2/prices/ETH-USD/spot`, public, no key). That endpoint returns a single price, not highs and lows, so ATR, the adverse-move history and `study` use daily candles from Coinbase Exchange's public candles endpoint instead. Kraken is an alternative provider. All of this is spot: a perpetual's mark price can differ, and liquidation uses mark.
- **Historical stats** use daily candles and overlapping windows. They indicate scale, not probability. The Coinbase adapter pages through 300-candle requests until it has the days you ask for; Kraken's endpoint returns at most 720 candles.
- **Equity** is your wallet balance excluding open PnL. `close` updates it with estimated fees and funding unless you pass the actual `--fees` / `--funding`. Your exchange statement is authoritative; this journal is a discipline tool.
- **Not verified live**: the Coinbase and Kraken adapters follow their documented response formats and are tested against fixtures, but were built in an environment with no access to those hosts. The spot parser reads only `data.amount`. Run `ethtracker price` and `ethtracker snapshot` first; if either fails, the error names the URL.

## Toward a bot

`risk.py` is pure (no I/O) on purpose: `plan_position` is the gate a bot would call before every order, and `mark_position` / `portfolio_status` are its kill-switch checks. What is missing, in order:

0. **Automatic exits**: `advise` already produces EXIT and TAKE_PARTIAL with reasons. Acting on them automatically needs steps 1 to 4 below, in particular a paper-trading period that shows the exit rules help rather than just trade less.
1. **Backtester**: `study` is the first cut (same sizing, liquidation, fee and funding functions, daily bars, one position). A proper one adds intraday candles for stop/target ordering, slippage, walk-forward validation and a rule set you have committed to before looking at results. Nothing should trade live that has not survived this.
2. **Strategy interface**: `(candles, state) -> intent`, with the risk gate between intent and order.
3. **Paper broker**, run for weeks against live prices.
4. **Exchange adapter**, only after 1 to 3, written against Coinbase's current Deribit-powered gateway rather than the legacy International Exchange API (see Venue above). Use API keys with trading but no withdrawal permission, keep them outside the repo, add a daily-loss kill-switch and a hard position cap enforced on the exchange side as well.

## Development

```
pytest
```
