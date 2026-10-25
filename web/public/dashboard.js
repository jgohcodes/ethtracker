import {
  usd, signedUsd, fixed, pct, signedPct, ageSeconds, freshness, humanAge, actionInfo, biasGlyph,
  priceLadder, isValidSymbol, describeInterval, finite, chance, eth, funding8h, PLAN_STATUS, TREND, CROWDING, lookup,
  SCORE, signedFixed, stateWord, stripSummary,
} from "./format.mjs";

const TOKEN_KEY = "ethtracker.viewToken";
const THEME_KEY = "ethtracker.theme";
const REFRESH_MS = 30_000;
const TICK_MS = 10_000;
const DEFAULT_SYMBOL = "COINBASE:ETHUSD";
const DERIVATIVE_SIGNALS = new Set(["funding", "oi_change", "oi_level", "long_liquidations", "short_liquidations"]);

const $ = (id) => document.getElementById(id);

// replaceChildren stringifies null as the text "null", so drop empty children first.
function fill(el, ...children) {
  el.replaceChildren(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
}

// Every value from a snapshot is inserted as text, never as markup, so nothing in it can inject HTML.
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("data-") || key.startsWith("aria-") || key === "role") el.setAttribute(key, value);
    else el[key] = value;
  }
  for (const child of children.flat()) if (child !== null && child !== undefined && child !== false) el.append(child);
  return el;
}

function readStorage(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeStorage(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* storage unavailable: the page still works for this visit */ }
}

let memoryToken = null;
const getToken = () => memoryToken ?? readStorage(TOKEN_KEY);
function setToken(token) {
  memoryToken = token;
  writeStorage(TOKEN_KEY, token);
}

let snap = null;
let mountedTheme = null;
let refreshTimer = null;

const badge = (tone, icon, text, large = false) =>
  h("span", { class: `badge${large ? " badge--lg" : ""}`, "data-tone": tone },
    h("span", { class: "badge__icon", "aria-hidden": "true", text: icon }), text);

function delta(value, text) {
  const sign = value > 0 ? "up" : value < 0 ? "down" : "flat";
  const icon = { up: "▲", down: "▼", flat: "•" }[sign];
  return h("span", { class: "delta", "data-sign": sign }, h("span", { class: "delta__icon", "aria-hidden": "true", text: icon }), text);
}

function effectiveTheme() {
  return document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}

// ---- chrome: banner, freshness, auth -------------------------------------------------------------

let bannerKind = null; // "freshness" banners may clear themselves; connection and waiting banners are cleared by a good load

function setBanner(tone, label, text, kind = "status") {
  const banner = $("banner");
  bannerKind = tone ? kind : null;
  if (!tone) {
    banner.hidden = true;
    banner.replaceChildren();
    return;
  }
  banner.dataset.tone = tone;
  banner.replaceChildren(h("div", { class: "banner__body" }, badge(tone, tone === "critical" ? "✕" : "!", label), h("p", { text })));
  banner.hidden = false;
}

function updateFreshness() {
  const chip = $("freshness");
  if (!snap) { chip.hidden = true; return; }
  const age = ageSeconds(snap.received_at ?? snap.generated_at);
  const interval = snap.refresh_seconds ?? 60;
  const level = freshness(age, interval);
  const style = {
    fresh: ["good", "✓", "Updated"], delayed: ["warning", "!", "Late"], stale: ["critical", "✕", "Stale"], unknown: ["muted", "?", "Unknown"],
  }[level];
  chip.replaceChildren(h("span", { class: "badge__icon", "aria-hidden": "true", text: style[1] }), `${style[2]} · ${humanAge(age)}`);
  chip.dataset.tone = style[0];
  chip.hidden = false;
  const fix = snap.publisher === "github-actions"
    ? "Check the \u201cDaily dashboard\u201d workflow in your repository's Actions tab."
    : "Check that `ethtracker publish` is running.";
  if (level === "stale") {
    setBanner("critical", "Stale", `The last update was ${humanAge(age)} and updates are expected ${describeInterval(interval)}, so positions and advice may be out of date. ${fix}`, "freshness");
  } else if (level === "delayed") {
    setBanner("warning", "Late", `The next update is overdue (expected ${describeInterval(interval)}). Positions and advice may lag the market. ${fix}`, "freshness");
  } else if (bannerKind === "freshness") {
    setBanner(null);
  }
}

function showLogin(message) {
  $("app").hidden = true;
  $("login").hidden = false;
  $("signout").hidden = true;
  $("freshness").hidden = true;
  $("price").replaceChildren();
  setBanner(null);
  const error = $("login-error");
  error.hidden = !message;
  error.textContent = message ?? "";
}

function showApp() {
  $("login").hidden = true;
  $("app").hidden = false;
  $("signout").hidden = false;
}

// ---- rendering -----------------------------------------------------------------------------------

function tile(label, value, sub) {
  return h("div", { class: "tile" },
    h("div", { class: "tile__label", text: label }),
    h("div", { class: "tile__value" }, value),
    sub ? h("div", { class: "tile__sub", text: sub }) : null);
}

function renderKpis(s) {
  const tiles = [];
  const eq = s.equity;
  if (eq) {
    tiles.push(tile("Equity", usd(eq.total), `wallet ${usd(eq.wallet)}`));
    tiles.push(tile("Open PnL", delta(eq.unrealized, signedUsd(eq.unrealized)), eq.wallet ? `${signedPct(eq.unrealized / eq.wallet)} of wallet` : ""));
  } else {
    tiles.push(tile("Equity", "Not set", "run `ethtracker equity <USD>`"));
  }
  const p = s.portfolio;
  if (s.plan) {
    tiles.push(...planTiles(s));
  } else if (p) {
    tiles.push(tile("Exposure", `${fixed(p.effective_leverage, 2)}x`, `gross ${usd(p.gross_notional, 0)} · net ${signedUsd(p.net_notional, 0)}`));
    const overRisk = p.flags.includes("OPEN_RISK_LIMIT");
    tiles.push(tile("Open risk to stops",
      [pct(p.open_risk_pct, 2), overRisk ? badge("warning", "!", "Over limit") : null],
      `${usd(p.open_risk_usd)} · limit ${fixed(s.config.max_open_risk_pct, 0)}%`));
    const st = p.stress;
    tiles.push(tile(`Stress: ${fixed(st.drop_pct, 0)}% move`,
      [`${pct(st.loss_pct, 0)} loss`, st.survives ? badge("good", "✓", "Holds") : badge("critical", "✕", "Breached")],
      st.headroom_notional === null ? "room to add: no limit set" : `room to add ${usd(st.headroom_notional, 0)}`));
    if (p.account_liq_price !== null && p.account_liq_price !== undefined && p.account_liq_price <= 0) {
      tiles.push(tile("Whole-book liquidation", "Not reachable", "equity exceeds exposure: the book cannot be liquidated at this size"));
    } else if (p.account_liq_price !== null && p.account_liq_price !== undefined) {
      const below = p.account_liq_price < s.price;
      const flagged = p.flags.includes("LIQ_DANGER") || p.flags.includes("LIQUIDATED");
      tiles.push(tile("Whole-book liquidation",
        [usd(p.account_liq_price), flagged ? badge("critical", "✕", "Near") : null],
        `${pct(Math.abs(1 - p.account_liq_price / s.price))} ${below ? "below" : "above"} the mark`));
    }
  }
  $("kpis").replaceChildren(...tiles);
}

function planTiles(s) {
  const plan = s.plan;
  const info = lookup(PLAN_STATUS, plan.status);
  const liq = plan.liq_price;
  const reachable = finite(liq) && liq > 0;
  const tiles = [];
  if (plan.qty > 0) {
    const p = s.portfolio;
    tiles.push(tile("Position", eth(plan.qty), p ? `${fixed(p.effective_leverage, 2)}x effective · ${usd(p.gross_notional, 0)}` : ""));
  }
  tiles.push(tile("Liquidation",
    [reachable ? usd(liq) : plan.qty > 0 ? "Not reachable" : "None", badge(info.tone, info.icon, info.label)],
    `floor ${usd(plan.floor, 0)}${reachable ? ` · ${pct(1 - liq / s.price)} under the mark` : ""}`));
  tiles.push(plan.cap_leverage === null
    ? tile("This week's add", "None", "price is at or under the floor")
    : tile("This week's add", eth(plan.weekly_qty), `${usd(plan.weekly_add, 0)} at up to ${fixed(plan.cap_leverage, 2)}x · ${usd(plan.weekly_notional, 0)}`));
  return tiles;
}

function metric(label, value) {
  return h("div", { class: "metric" }, h("div", { class: "metric__label", text: label }), h("div", { class: "metric__value" }, value));
}

const CASE_VISIBLE = 4;

function caseList(title, items, glyph, empty) {
  const row = (text) => h("li", {}, h("span", { class: "list__glyph", "aria-hidden": "true", text: items.length ? glyph : "" }), h("span", { text }));
  const shown = (items.length ? items : [empty]).slice(0, CASE_VISIBLE);
  const rest = items.slice(CASE_VISIBLE);
  return h("div", {},
    h("h3", { text: title }),
    h("ul", { class: "list" }, shown.map(row)),
    rest.length ? h("details", { class: "more" }, h("summary", { text: `${rest.length} more` }), h("ul", { class: "list" }, rest.map(row))) : null);
}

function lotsTable(lots) {
  return h("details", { class: "more" },
    h("summary", { text: `${lots.length} buys` }),
    h("div", { class: "scroll" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["#", "Date", "ETH", "Price", "Open PnL"].map((t, i) => h("th", { class: i > 1 ? "num" : "", text: t })))),
      h("tbody", {}, lots.map((lot) => h("tr", {},
        h("td", { text: String(lot.id) }),
        h("td", { text: (lot.opened_at ?? "").slice(0, 10) }),
        h("td", { class: "num", text: fixed(lot.qty, 4) }),
        h("td", { class: "num", text: usd(lot.entry) }),
        h("td", { class: "num" }, delta(lot.unrealized, signedUsd(lot.unrealized)))))))));
}

function positionCard(p, planned) {
  const info = actionInfo(p.action);
  const label = info.label + (p.action === "TAKE_PARTIAL" ? ` ${pct(p.suggested_fraction, 0)}` : "");
  const tags = [`leverage ${fixed(p.leverage, 0)}x`, `net vote ${p.net_score > 0 ? "+" : ""}${p.net_score}`, `${p.risk_flags} risk flag${p.risk_flags === 1 ? "" : "s"}`];
  if (p.deep_itm) tags.unshift("deep in the money");
  for (const flag of p.flags) tags.push(flag);

  const closing = p.suggested_fraction > 0 && (p.action === "TAKE_PARTIAL" || p.action === "EXIT");
  return h("article", { class: "pos", "aria-label": `Position ${p.id}` },
    h("div", { class: "pos__head" },
      h("div", {},
        h("div", { class: "pos__title", text: p.lots
          ? `${p.side.toUpperCase()} ${fixed(p.qty, 4)} ETH @ ${usd(p.entry)} average`
          : `#${p.id} ${p.side.toUpperCase()} ${fixed(p.qty, 4)} ETH @ ${usd(p.entry)}` }),
        h("div", { class: "pos__tags" }, tags.map((t) => h("span", { class: "tag", text: t })))),
      badge(info.tone, info.icon, label, true)),
    h("div", { class: "metrics" },
      metric("Mark", usd(p.mark)),
      metric("Move from entry", delta(p.move, signedPct(p.move))),
      metric("Open PnL", delta(p.unrealized, signedUsd(p.unrealized))),
      metric(planned ? "Floor" : "Stop", `${usd(p.stop)} (${pct(p.dist_to_stop)} away)`),
      planned ? null : metric("Target", p.target === null ? "none" : usd(p.target)),
      metric(p.lots ? "Since the first buy" : "Held", p.max_days === null || p.max_days === undefined
        ? `${fixed(p.held_days, 0)} days` : `${fixed(p.held_days, 1)} / ${p.max_days} days`),
      metric("Funding so far", usd(p.funding_accrued)),
      p.liq_price === null || p.liq_price === undefined ? null : metric("Liquidation", `${usd(p.liq_price)} (${pct(p.dist_to_liq)} away)`)),
    h("div", { class: "advice" },
      h("div", { class: "advice__line" }, h("strong", { text: "Why: " }), p.reasons.join(" ")),
      p.suggested_stop === null || p.suggested_stop === undefined ? null
        : h("div", { class: "advice__line" }, h("strong", { text: "Stop: " }), `move to ${usd(p.suggested_stop)} (now ${usd(p.stop)})`),
      closing ? h("div", { class: "advice__line" }, h("strong", { text: "Size: " }), `close ${pct(p.suggested_fraction, 0)} of the position (${fixed(p.qty * p.suggested_fraction, 4)} ETH)`) : null,
      h("div", { class: "cases" },
        caseList("Bull case", p.bull, "▲", "Nothing supportive right now."),
        caseList("Bear case", p.bear, "▼", "Nothing flagged.")),
      p.lots ? lotsTable(p.lots) : p.note ? h("div", { class: "muted", text: `Note: ${p.note}` }) : null));
}

function renderPositions(s) {
  const root = $("positions");
  if (!s.positions.length) {
    root.replaceChildren(h("div", { class: "empty", text: s.plan
      ? "No position yet. Record your first buy with `ethtracker add --usd <cash> --price <fill>`, then `ethtracker publish --once`."
      : "No open positions. Record one with `ethtracker open` and it appears here on the next publish." }));
    return;
  }
  root.replaceChildren(...s.positions.map((p) => positionCard(p, Boolean(s.plan))));
}

function planLine(s) {
  const plan = s.plan;
  if (plan.status === "above the floor") {
    return `Liquidation ${usd(plan.liq_price)} is above your ${usd(plan.floor, 0)} floor. Bring it back before adding: ` +
      `sell ${eth(plan.cut_qty)} or deposit ${usd(plan.margin_needed, 0)}.`;
  }
  if (plan.cap_leverage === null) {
    return `Price is at or under your ${usd(plan.floor, 0)} floor: nothing can be added without lifting the liquidation price above it.`;
  }
  if (plan.status === "no position") {
    return `No position recorded yet. At ${usd(s.price)}, new cash can carry up to ${fixed(plan.cap_leverage, 2)}x and keep liquidation at ${usd(plan.floor, 0)}.`;
  }
  return `Liquidation ${usd(plan.liq_price)} is ${plan.status === "at the floor" ? "at" : "under"} your ${usd(plan.floor, 0)} floor. ` +
    `This week's ${usd(plan.weekly_add, 0)} buys up to ${eth(plan.weekly_qty)} at the mark.`;
}

function renderPlan(s) {
  const plan = s.plan;
  $("plan-card").hidden = !plan;
  if (!plan) { $("plan").replaceChildren(); return; }
  const info = lookup(PLAN_STATUS, plan.status);
  const basis = s.odds?.in_pullback ? "pullback day (15%+ under the 30-day high, like today)" : "day";
  const rows = plan.ladder.map((r) => h("tr", {},
    h("td", { text: `−${pct(r.depth, 0)}` }),
    h("td", { class: "num", text: usd(r.price, 0) }),
    h("td", { class: "num", text: r.add_qty > 0 ? fixed(r.add_qty, 4) : "none" }),
    h("td", { class: "num", text: usd(r.add_notional, 0) }),
    h("td", { class: "num", text: finite(r.cap_leverage) ? `${fixed(r.cap_leverage, 2)}x` : "n/a" }),
    h("td", { class: "num", text: chance(r.filled) }),
    h("td", { class: "num", text: chance(r.held) }),
    h("td", { class: "num", text: chance(r.higher_365) })));
  fill($("plan"),
    h("div", { class: "lead" }, badge(info.tone, info.icon, info.label, true), h("p", { text: planLine(s) })),
    h("div", { class: "metrics" },
      metric("Floor", usd(plan.floor, 0)),
      metric("Weekly add", usd(plan.weekly_add, 0)),
      metric("New cash can carry", plan.cap_leverage === null ? "nothing" : `${fixed(plan.cap_leverage, 2)}x`),
      metric("This week at the mark", plan.cap_leverage === null ? "none" : `${eth(plan.weekly_qty)} · ${usd(plan.weekly_notional, 0)}`),
      metric("Room without new cash", eth(plan.room_qty)),
      plan.set_at ? metric("Plan set", plan.set_at.slice(0, 10)) : null),
    h("h3", { class: "sub", text: "If price falls first: this week's add as a buy order under the mark" }),
    h("div", { class: "scroll" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["Order", "Price", "ETH", "Notional", "Max lev.", "Filled 30d", "Low held", "Up 1y on"]
        .map((t, i) => h("th", { class: i ? "num" : "", text: t })))),
      h("tbody", {}, rows))),
    h("p", { class: "note", text:
      `The nearer the floor, the more leverage new cash can carry, so each dollar buys more lower down. ` +
      `Fill and outcome rates: from every past ${basis} in ETH's history, an order this far under the close, good for 30 days. ` +
      `"Low held": no further 10% fall within 30 days of the fill. In this repository's studies, adding every week beat ` +
      `saving cash for a signal in every window tested, and choosing the day within a few days barely changed results: the schedule matters more than the order price.` }));
}

function renderFloorRisk(s) {
  const r = s.floor_risk;
  $("risk-card").hidden = !r;
  if (!r) { $("floor-risk").replaceChildren(); return; }
  const fall = finite(r.distance) ? 1 - 1 / (1 + r.distance) : null;
  const hist = r.history;
  fill($("floor-risk"),
    h("div", { class: "metrics" },
      metric("Fall to the floor", finite(fall) ? `−${pct(fall)}` : "n/a"),
      metric("Volatility forecast", `${pct(r.vol_forecast, 0)} a year`),
      metric("30-day realised", `${pct(r.vol_30d, 0)} a year`),
      metric("Distance", finite(r.sigmas) ? `${fixed(r.sigmas, 2)} one-year σ` : "n/a"),
      metric("Touch within 30 days", chance(r.touch_30d)),
      metric("Touch within a year", chance(r.touch_1y))),
    hist ? h("p", { class: "note", text:
      `History: from days ${pct(hist.from_depth, 0)} or more under the all-time high (like today), ${hist.starts} start days in ` +
      `${hist.drawdowns} separate drawdowns. A fall this far followed within a year ${pct(hist.within_1y, 0)} of the time, ` +
      `within four years ${pct(hist.within_4y, 0)}. ${hist.drawdowns} drawdowns is a small sample: low confidence.` }) : null,
    h("p", { class: "note", text:
      "Touch odds: a random walk with no drift at the forecast volatility (RiskMetrics EWMA), so no view on direction. " +
      "Volatility clusters, so these move with the market. Reaching the floor means liquidation of the whole book." }));
}

function renderMarket(s) {
  const t = s.trend, c = s.crowding;
  $("market-card").hidden = !t && !c;
  if (!t && !c) { $("market").replaceChildren(); return; }
  const tr = t ? lookup(TREND, t.label) : null;
  const cr = c ? lookup(CROWDING, c.label) : null;
  const side = (above) => (above ? "above" : "below");
  fill($("market"),
    t ? h("div", { class: "group" },
      h("div", { class: "lead" }, h("h3", { text: "Trend" }), badge(tr.tone, tr.icon, tr.label)),
      h("p", { text: `Close ${usd(t.close, 0)}: ${side(t.above_50)} the 50-day (${usd(t.sma50, 0)}), ${side(t.above_200)} the 200-day (${usd(t.sma200, 0)}); ${signedPct(t.return_28d)} over 28 days.` }),
      h("p", { class: "note", text: "Trend (time-series momentum) is the return signal with the most out-of-sample support in crypto research. In a downtrend, a fall to the floor is likelier; the weekly add still runs." })) : null,
    h("div", { class: "group" },
      h("div", { class: "lead" }, h("h3", { text: "Leverage crowding" }), cr ? badge(cr.tone, cr.icon, cr.label) : h("span", { class: "muted", text: "no data" })),
      c ? h("p", { text: `Funding ${funding8h(c.funding_8h)}: ${pct(c.percentile, 0)} of the last ${c.days} days were at or below it (median ${funding8h(c.median_8h)}).` })
        : h("p", { class: "muted", text: "Deribit's funding history was unavailable for this snapshot." }),
      h("p", { class: "note", text: "High funding means crowded longs: a fall would force more selling. It measures fragility, not timing: in this repository's short study, shorting overheating signals (high funding among them) did worse than holding." })));
}

function renderOdds(s) {
  const o = s.odds;
  $("odds-card").hidden = !o;
  if (!o) { $("odds").replaceChildren(); return; }
  const rows = o.signals.map((r) => {
    let fired = "n/a";
    if (!r.baseline) fired = r.last_fired === null ? "never" : `${r.days_ago}d ago`;
    return h("tr", { "data-kind": r.baseline ? "baseline" : r.active ? "active" : "" },
      h("td", {}, r.name, r.active ? [" ", badge("warning", "!", "Fired")] : null),
      h("td", { class: "nowrap", text: r.baseline ? "baseline" : fired, title: r.last_fired ?? undefined }),
      h("td", { class: "num", text: String(r.events) }),
      h("td", { class: "num", text: finite(r.held) ? `${chance(r.held)} (≥${chance(r.held_low)})` : "n/a" }),
      h("td", { class: "num", text: chance(r.higher_365) }));
  });
  fill($("odds"),
    h("p", { text: `ETH is ${o.in_pullback ? "" : "not "}in a pullback (${pct(o.pullback_depth, 0)}+ under its 30-day high). ` +
      `For each signal, how often the low then held: no further ${pct(o.further_fall, 0)} fall within ${o.held_days} days of the close. ` +
      "Compare each with the baselines. None reaches 80%." }),
    h("div", { class: "scroll" }, h("table", { class: "tbl odds" },
      h("thead", {}, h("tr", {}, ["Signal", "Last fired", "Times", "Low held (95% low)", "Up 1y on"].map((t, i) => h("th", { class: i > 1 ? "num" : "", text: t })))),
      h("tbody", {}, rows))),
    h("p", { class: "note", text: "Information, not instructions: under the adding plan these never change the weekly add." }));
}

// ---- the exchange's own view of the position ---------------------------------------------------------

function renderAccount(s) {
  const a = s.account;
  $("account-card").hidden = !a;
  if (!a) { $("account").replaceChildren(); return; }
  const floor = s.plan?.floor;
  const estimate = s.plan?.liq_price;
  const when = (a.fetched_at ?? "").replace("T", " ").replace("Z", " UTC");
  const positions = (a.positions ?? []).map((p) => {
    const liq = p.liquidation;
    const vsFloor = finite(liq) && finite(floor)
      ? (liq > floor * 1.001 ? badge("critical", "✕", "Above your floor") : badge("good", "✓", "At or under your floor")) : null;
    const gap = finite(liq) && finite(estimate) && estimate > 0 ? liq - estimate : null;
    return h("div", { class: "group" },
      h("div", { class: "lead" }, h("h3", { text: `${p.product ?? "ETH perpetual"}: ${p.side} ${fixed(Math.abs(p.size ?? NaN), 4)} ETH` }), vsFloor),
      h("div", { class: "metrics" },
        metric("Entry (average)", usd(p.entry)),
        metric("Mark", usd(p.mark)),
        metric("Liquidation, per Coinbase", usd(liq)),
        gap === null ? null : metric("Versus this tool's estimate", `${gap >= 0 ? "+" : "−"}${usd(Math.abs(gap))}`),
        metric("Open PnL", delta(p.unrealized, signedUsd(p.unrealized))),
        p.leverage === null || p.leverage === undefined ? null : metric("Leverage", `${fixed(p.leverage, 2)}x`),
        p.notional === null || p.notional === undefined ? null : metric("Position value", usd(p.notional, 0))));
  });
  const sm = a.summary;
  const summary = sm ? h("div", { class: "metrics" },
    sm.collateral === null || sm.collateral === undefined ? null : metric("Collateral", usd(sm.collateral)),
    sm.liquidation_buffer === null || sm.liquidation_buffer === undefined ? null : metric("Liquidation buffer", usd(sm.liquidation_buffer)),
    sm.unrealized === null || sm.unrealized === undefined ? null : metric("Open PnL, all", delta(sm.unrealized, signedUsd(sm.unrealized)))) : null;
  const fills = (a.fills ?? []).length ? h("div", {},
    h("h3", { class: "sub", text: "Recent ETH perpetual fills" }),
    h("div", { class: "scroll" }, h("table", { class: "tbl" },
      h("thead", {}, h("tr", {}, ["Time (UTC)", "Side", "ETH", "Price", "Fee"].map((t, i) => h("th", { class: i > 1 ? "num" : "", text: t })))),
      h("tbody", {}, a.fills.map((f) => h("tr", {},
        h("td", { text: (f.time ?? "").slice(0, 16).replace("T", " ") }), h("td", { text: f.side }),
        h("td", { class: "num", text: fixed(f.size, 4) }), h("td", { class: "num", text: usd(f.price) }),
        h("td", { class: "num", text: usd(f.fee) }))))))) : null;
  fill($("account"),
    positions.length ? positions : h("p", { class: "muted", text:
      "Coinbase's API returned no ETH perpetual position. If you hold one, it may sit on the Deribit-powered venue this key cannot see yet." }),
    summary, fills,
    h("p", { class: "note", text: `Read from Coinbase with your view-only key at ${when}. These are the exchange's own numbers; ` +
      "where they differ from this tool's, trust these and tell me the gap so the plan's margin rate can be corrected." }));
}

// ---- the signal score -----------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== undefined && value !== null) el.setAttribute(key, value);
  for (const child of children.flat()) if (child) el.append(child);
  return el;
}

const SIGNAL_NAMES = {
  rsi: "RSI (14 days)", rsi_rollover: "RSI rolling over", extension: "Stretch from the 20-day average",
  volatility: "Volatility (risk only)", avwap_low: "AVWAP from the 365-day low", avwap_year: "AVWAP from the year open",
  poc: "Volume profile (365 days)", funding: "Funding level", oi_change: "Open interest, 24 hours",
  oi_level: "Open interest level", long_liquidations: "Long liquidations", short_liquidations: "Short liquidations",
};
const CELL = 8, GAP = 3, STRIP_H = 22;
const stripWidth = (n) => n * (CELL + GAP) - GAP;

// Up bar, down bar or a dot on the midline: the sign reads from position and shape, colour only adds to it.
function strip(values) {
  const mid = STRIP_H / 2;
  const marks = values.map((v, i) => {
    const x = i * (CELL + GAP);
    const now = i === values.length - 1 ? " strip__now" : "";
    if (v === null || v === undefined) return null;
    if (v > 0) return svg("rect", { x, y: 2, width: CELL, height: mid - 3, rx: 1.5, class: `strip__pos${now}` });
    if (v < 0) return svg("rect", { x, y: mid + 1, width: CELL, height: mid - 3, rx: 1.5, class: `strip__neg${now}` });
    return svg("circle", { cx: x + CELL / 2, cy: mid, r: 2, class: `strip__flat${now}` });
  });
  const w = stripWidth(values.length);
  return svg("svg", { viewBox: `0 0 ${w} ${STRIP_H}`, width: w, height: STRIP_H, role: "img", "aria-label": stripSummary(values), class: "strip" },
    svg("line", { x1: 0, x2: w, y1: mid, y2: mid, class: "strip__axis" }), marks);
}

// The strip's header: where 14, 7 and 3 days ago and today fall.
function stripAxis(n) {
  const w = stripWidth(n);
  const at = (back) => (n - 1 - back) * (CELL + GAP) + CELL / 2;
  return svg("svg", { viewBox: `0 0 ${w} 12`, width: w, height: 12, "aria-hidden": "true", class: "strip strip--axis" },
    [[14, "14d"], [7, "7d"], [3, "3d"], [0, "now"]].filter(([b]) => b < n).map(([b, t]) =>
      svg("text", { x: at(b), y: 10, "text-anchor": b === 0 ? "end" : b === 14 ? "start" : "middle" }, document.createTextNode(t))));
}

function scoreLine(history) {
  const W = 360, H = 64, pad = 4;
  const n = history.length;
  if (n < 2) return null;
  const x = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
  const y = (v) => pad + ((1 - v) / 2) * (H - 2 * pad);
  const points = history.map((p, i) => `${x(i).toFixed(1)},${y(p.score).toFixed(1)}`).join(" ");
  const marks = [14, 7, 3].filter((b) => b < n).map((b) =>
    svg("circle", { cx: x(n - 1 - b), cy: y(history[n - 1 - b].score), r: 2.5, class: "spark__mark" }));
  const last = history[n - 1].score;
  return svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "spark", role: "img",
    "aria-label": `Score over the last ${n} days: from ${signedFixed(history[0].score)} to ${signedFixed(last)}` },
  svg("line", { x1: pad, x2: W - pad, y1: y(0), y2: y(0), class: "spark__zero" }),
  svg("line", { x1: pad, x2: W - pad, y1: y(0.2), y2: y(0.2), class: "spark__band" }),
  svg("line", { x1: pad, x2: W - pad, y1: y(-0.2), y2: y(-0.2), class: "spark__band" }),
  svg("polyline", { points, class: "spark__line" }), marks,
  svg("circle", { cx: x(n - 1), cy: y(last), r: 4, class: last > 0 ? "spark__now spark__now--pos" : last < 0 ? "spark__now spark__now--neg" : "spark__now" }));
}

function stateCell(v) {
  const glyph = v === null || v === undefined ? "" : v > 0 ? "▲" : v < 0 ? "▼" : "•";
  return h("td", { class: "state", "data-sign": v > 0 ? "up" : v < 0 ? "down" : "flat" },
    h("span", { class: "state__glyph", "aria-hidden": "true", text: glyph }), stateWord(v));
}

function renderScore(s) {
  const sc = s.score;
  const show = sc && sc.value !== null && sc.value !== undefined;
  $("score-card").hidden = !show;
  if (!show) { $("score").replaceChildren(); return; }
  const info = lookup(SCORE, sc.label);
  const n = sc.components[0]?.strip.length ?? 15;
  const cal = sc.calibration;
  const ago = [["14", "14 days ago"], ["7", "7 days ago"], ["3", "3 days ago"]]
    .map(([k, t]) => h("span", { class: "ago" }, h("span", { class: "muted", text: `${t} ` }), signedFixed(sc.ago[k])));
  const lead = h("div", { class: "score__lead" },
    h("div", { class: "score__value" }, h("span", { class: "score__number", text: signedFixed(sc.value) }), badge(info.tone, info.icon, info.label, true)),
    h("div", { class: "score__chart" }, scoreLine(sc.history), h("div", { class: "score__ago" }, ago)));
  // the institutional era leads; the full history follows, because the large early swings skew its averages
  const recentLine = cal?.same_recent?.days
    ? `Since ${cal.recent_from.slice(0, 4)}, days scored ${cal.bucket.toLowerCase()} (${signedFixed(cal.range[0], 1)} to ${signedFixed(cal.range[1], 1)}; ` +
      `${cal.same_recent.days.toLocaleString("en-US")} days in ${cal.same_recent.episodes} separate stretches): ETH was higher ${cal.horizon} days later ` +
      `${chance(cal.same_recent.higher)} of the time (median ${signedPct(cal.same_recent.median)}) and fell 10%+ within them ${chance(cal.same_recent.fell)} ` +
      `of the time, against ${chance(cal.all_recent.higher)}, ${signedPct(cal.all_recent.median)} and ${chance(cal.all_recent.fell)} for every day since then. `
    : "";
  const meaning = cal && cal.same.days ? h("p", { class: "score__meaning", text: recentLine +
    `All years since ${cal.from.slice(0, 4)}: higher ${chance(cal.same.higher)}, median ${signedPct(cal.same.median)} ` +
    `(every day: ${chance(cal.all.higher)}, ${signedPct(cal.all.median)}).` }) : null;

  const scored = sc.components.map((c) => h("tr", {},
    h("td", {}, h("div", { text: c.name }), h("div", { class: "sub-line" }, h("span", { class: "tag", text: c.evidence === "failed" ? "failed the check" : `${c.evidence} evidence` }), " ", c.reads)),
    h("td", { class: "num", "data-label": "weight", text: `${Math.round(c.weight * 100)}%` }),
    h("td", { class: "strip-cell" }, strip(c.strip)),
    stateCell(c.state),
    h("td", { class: "num", "data-label": "adds", text: c.contribution === null ? "n/a" : signedFixed(c.contribution) })));
  const others = (sc.unscored ?? []).map((u) => h("tr", {},
    h("td", {}, h("div", { text: SIGNAL_NAMES[u.name] ?? u.name }), h("div", { class: "sub-line", text: u.headline }),
      u.risk ? badge("warning", "!", "Risk") : null),
    h("td", { class: "num", "data-label": "weight", text: "0%" }),
    h("td", { class: "strip-cell" }, strip(u.strip)),
    stateCell(u.state),
    h("td", { class: "num", text: "" })));
  const head = h("thead", {}, h("tr", {},
    h("th", { text: "Signal" }), h("th", { class: "num", text: "Weight" }),
    h("th", { class: "strip-cell" }, h("span", { class: "sr", text: "Last 15 days" }), stripAxis(n)), h("th", { text: "Now" }), h("th", { class: "num", text: "Adds" })));
  const legend = h("p", { class: "note score__legend" },
    `Strips: the last ${n} days, oldest on the left, today outlined. A bar up is positive, a bar down negative, a dot neutral. `,
    "\u201cAdds\u201d is each signal's share of today's score.");
  fill($("score"),
    lead, meaning, legend,
    h("div", { class: "scroll" }, h("table", { class: "tbl score__table" }, head,
      h("tbody", {}, scored),
      others.length ? h("tbody", { class: "score__unscored" },
        h("tr", {}, h("td", { colSpan: 5, class: "score__group", text: "Shown, not scored: no published evidence of a directional edge" })), others) : null)),
    h("details", { class: "more score__how" }, h("summary", { text: "How the score is built" }),
      h("p", { class: "note", text:
        "Each scored signal reads +1, 0 or −1 every day; the score is their weighted average, from −1 to +1 (a part with no data, such as funding before its history starts, drops out and the rest are rescaled). " +
        "The weights were set from the research before looking at ETH's results: three readings of momentum and trend, the one effect with repeated out-of-sample support in crypto, carry 86%; volume pressure, a plausible mechanism with no published test, carries 14%. " +
        "Then the check on ETH's history, 2016 to 2026: every scored part pointed the right way in both halves (before and from 2021), but leverage crowding pointed the wrong way in both, so it was set to zero. " +
        "The edge is modest: the score's rank correlation with the next 30 days was +0.14, against about \u00b10.08 from chance alone, and +0.10 since 2021. " +
        "RSI, stretch, anchored VWAP, the volume profile and open interest are shown with their history but carry no weight. " +
        "The score is information, not an instruction: under the adding plan it never changes the weekly add." }),
      h("p", { class: "note" }, "Sources: ",
        h("a", { href: "https://ideas.repec.org/a/oup/rfinst/v34y2021i6p2689-2727..html", target: "_blank", rel: "noopener noreferrer", text: "Liu & Tsyvinski (2021), Review of Financial Studies" }),
        "; ",
        h("a", { href: "https://profiles.wustl.edu/en/publications/learning-and-predictability-via-technical-analysis-evidence-from-/", target: "_blank", rel: "noopener noreferrer", text: "Detzel et al. (2021), Financial Management" }),
        ". Checked on ETH's own history with `ethtracker score-study`.")));
}

function signalRow(g) {
  const word = g.bias > 0 ? "bullish" : g.bias < 0 ? "bearish" : "neutral";
  const meta = [];
  if (g.risk) meta.push(badge("warning", "!", "Risk"));
  if (g.unvalidated) meta.push(h("span", { class: "tag", text: "unvalidated" }));
  return h("li", { class: "sig" },
    h("span", { class: "sig__glyph", role: "img", "aria-label": word, text: biasGlyph(g.bias) }),
    h("span", { class: "sig__text" }, g.headline, meta.length ? h("span", { class: "sig__meta" }, meta) : null));
}

function renderSignals(s) {
  const technical = s.signals.filter((g) => !DERIVATIVE_SIGNALS.has(g.name));
  const derivatives = s.signals.filter((g) => DERIVATIVE_SIGNALS.has(g.name));
  const groups = [["Technical", technical], ["Derivatives", derivatives]].filter(([, list]) => list.length);
  const votes = Object.entries(s.votes ?? {}).filter(([, vote]) => vote !== 0);
  const total = Object.values(s.votes ?? {}).reduce((sum, vote) => sum + vote, 0);
  fill($("signals"),
    groups.map(([title, list]) => h("div", { class: "group" }, h("h3", { text: title }), h("ul", {}, list.map(signalRow)))),
    h("p", { class: "votes", text: `Net by family (each votes once): ${votes.length ? votes.map(([name, vote]) => `${name} ${biasGlyph(vote)}`).join(" · ") : "none voting"} · total ${total > 0 ? "+" : ""}${total}` }),
    s.notes.length ? h("ul", { class: "notes" }, s.notes.map((note) => h("li", { text: note }))) : null);
}

function renderLadder(s) {
  const rows = priceLadder(s);
  $("ladder").replaceChildren(h("div", { class: "scroll" }, h("table", { class: "ladder" },
    h("thead", {}, h("tr", {}, h("th", { text: "Level" }), h("th", { class: "num", text: "Price" }), h("th", { class: "num", text: "From mark" }))),
    h("tbody", {}, rows.map((row) => h("tr", { "data-kind": row.kind },
      h("td", { text: row.label }),
      h("td", { class: "num", text: usd(row.price) }),
      h("td", { class: "num", text: row.kind === "mark" ? "" : signedPct(row.distance) })))))));
}

function renderClosed(s) {
  const root = $("closed");
  if (!s.closed.length) {
    root.replaceChildren(h("p", { class: "muted", text: "None yet." }));
    return;
  }
  root.replaceChildren(h("div", { class: "scroll" }, h("table", { class: "tbl" },
    h("thead", {}, h("tr", {}, ["#", "Side", "Entry", "Exit", "Realized", "R", "Closed"].map((t, i) => h("th", { class: i > 1 && i < 6 ? "num" : "", text: t })))),
    h("tbody", {}, s.closed.map((c) => h("tr", {},
      h("td", { text: String(c.id) }), h("td", { text: c.side }),
      h("td", { class: "num", text: usd(c.entry) }), h("td", { class: "num", text: usd(c.exit) }),
      h("td", { class: "num" }, delta(c.realized_pnl, signedUsd(c.realized_pnl))),
      h("td", { class: "num", text: c.r === null ? "n/a" : `${c.r > 0 ? "+" : ""}${fixed(c.r, 2)}R` }),
      h("td", { text: (c.closed_at ?? "").slice(0, 10) })))))));
}

function renderHeader(s) {
  const basis = s.basis === "close" && s.close_date ? `signals on the ${s.close_date} daily close` : "signals include today's unfinished candle";
  fill($("price"), usd(s.price), " ",
    s.price_1d === null || s.price_1d === undefined ? null : delta(s.price_1d, `${signedPct(s.price_1d)} vs last daily close`),
    h("span", { class: "bar__basis", text: basis }));
}

function renderFoot(s) {
  const by = s.publisher === "github-actions" ? "the daily GitHub Actions job" : s.publisher === "local" ? "ethtracker on your machine" : "ethtracker";
  $("foot").textContent =
    `Computed by ${by} at ${s.generated_at} (${describeInterval(s.refresh_seconds ?? 60)} updates); this page only displays it. ` +
    `Candles from ${s.source || "unknown"}; derivatives signals use ${s.config.instrument} and are unvalidated. ` +
    "The chart is TradingView's embedded widget: it cannot draw your levels, which is what the price ladder is for.";
}

function mountChart() {
  const wanted = effectiveTheme();
  if (mountedTheme === wanted) return;
  mountedTheme = wanted;
  const requested = new URLSearchParams(location.search).get("symbol");
  const symbol = isValidSymbol(requested) ? requested : DEFAULT_SYMBOL;
  const script = document.createElement("script");
  script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
  script.async = true;
  script.textContent = JSON.stringify({
    autosize: true, symbol, interval: "D", timezone: "Etc/UTC", theme: wanted, style: "1", locale: "en",
    allow_symbol_change: false, studies: ["RSI@tv-basicstudies"],
  });
  const holder = h("div", { class: "tradingview-widget-container" }, h("div", { class: "tradingview-widget-container__widget" }));
  holder.append(script);
  // sits behind the widget and shows only if it never paints (blocked network, ad blocker)
  fill($("chart"), h("p", { class: "chart__fallback", text: "Loading the TradingView chart. If this stays blank, something is blocking s3.tradingview.com (a firewall or an ad blocker)." }), holder);
}

function render(s) {
  showApp();
  renderHeader(s);
  renderKpis(s);
  renderPlan(s);
  renderAccount(s);
  renderScore(s);
  renderFloorRisk(s);
  renderMarket(s);
  renderPositions(s);
  renderSignals(s);
  renderOdds(s);
  renderLadder(s);
  renderClosed(s);
  renderFoot(s);
  mountChart();
  updateFreshness();
}

// ---- loading -------------------------------------------------------------------------------------

async function load() {
  const token = getToken();
  if (!token) { showLogin(); return; }
  $("app").classList.add("refreshing");
  try {
    const response = await fetch("/api/snapshot", { headers: { Authorization: `Bearer ${token}` }, cache: "no-store" });
    if (response.status === 401) {
      setToken(null);
      snap = null;
      showLogin("That token was not accepted. Check DASHBOARD_TOKEN on the deployment.");
      return;
    }
    if (response.status === 404) {
      showApp();
      snap = null;
      $("freshness").hidden = true;
      setBanner("warning", "Waiting", "No snapshot has been published yet. Run `ethtracker publish` on the machine that holds your positions.");
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    snap = await response.json();
    setBanner(null);
    render(snap);
  } catch {
    showApp();
    setBanner("critical", "Offline", snap
      ? "Cannot reach the dashboard API. Showing the last snapshot, which is ageing."
      : "Cannot reach the dashboard API.");
    updateFreshness();
  } finally {
    $("app").classList.remove("refreshing");
  }
}

function schedule() {
  clearInterval(refreshTimer);
  refreshTimer = setInterval(() => { if (!document.hidden) load(); }, REFRESH_MS);
}

function init() {
  $("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const value = $("token").value.trim();
    if (!value) return;
    setToken(value);
    $("token").value = "";
    load();
  });
  $("signout").addEventListener("click", () => {
    setToken(null);
    snap = null;
    for (const id of ["kpis", "plan", "account", "score", "floor-risk", "market", "odds", "positions", "signals", "ladder", "closed", "chart"]) $(id).replaceChildren();
    mountedTheme = null;
    showLogin();
  });
  $("theme").addEventListener("click", () => {
    const order = [null, "light", "dark"];
    const current = document.documentElement.dataset.theme ?? null;
    const next = order[(order.indexOf(current) + 1) % order.length];
    if (next) document.documentElement.dataset.theme = next;
    else delete document.documentElement.dataset.theme;
    writeStorage(THEME_KEY, next);
    $("theme").textContent = `Theme: ${next ?? "system"}`;
    if (snap) mountChart();
  });
  $("theme").textContent = `Theme: ${document.documentElement.dataset.theme ?? "system"}`;
  document.addEventListener("visibilitychange", () => { if (!document.hidden && getToken()) load(); });
  setInterval(updateFreshness, TICK_MS);
  schedule();
  load();
}

init();
