import {
  usd, signedUsd, fixed, pct, signedPct, ageSeconds, freshness, humanAge, actionInfo, biasGlyph,
  priceLadder, isValidSymbol, describeInterval,
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
  if (p) {
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

function positionCard(p) {
  const info = actionInfo(p.action);
  const label = info.label + (p.action === "TAKE_PARTIAL" ? ` ${pct(p.suggested_fraction, 0)}` : "");
  const tags = [`leverage ${fixed(p.leverage, 0)}x`, `net vote ${p.net_score > 0 ? "+" : ""}${p.net_score}`, `${p.risk_flags} risk flag${p.risk_flags === 1 ? "" : "s"}`];
  if (p.deep_itm) tags.unshift("deep in the money");
  for (const flag of p.flags) tags.push(flag);

  const closing = p.suggested_fraction > 0 && (p.action === "TAKE_PARTIAL" || p.action === "EXIT");
  return h("article", { class: "pos", "aria-label": `Position ${p.id}` },
    h("div", { class: "pos__head" },
      h("div", {},
        h("div", { class: "pos__title", text: `#${p.id} ${p.side.toUpperCase()} ${fixed(p.qty, 4)} ETH @ ${usd(p.entry)}` }),
        h("div", { class: "pos__tags" }, tags.map((t) => h("span", { class: "tag", text: t })))),
      badge(info.tone, info.icon, label, true)),
    h("div", { class: "metrics" },
      metric("Mark", usd(p.mark)),
      metric("Move from entry", delta(p.move, signedPct(p.move))),
      metric("Open PnL", delta(p.unrealized, signedUsd(p.unrealized))),
      metric("Stop", `${usd(p.stop)} (${pct(p.dist_to_stop)} away)`),
      metric("Target", p.target === null ? "none" : usd(p.target)),
      metric("Held", `${fixed(p.held_days, 1)} / ${p.max_days} days`),
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
      p.note ? h("div", { class: "muted", text: `Note: ${p.note}` }) : null));
}

function renderPositions(s) {
  const root = $("positions");
  if (!s.positions.length) {
    root.replaceChildren(h("div", { class: "empty", text: "No open positions. Record one with `ethtracker open` and it appears here on the next publish." }));
    return;
  }
  root.replaceChildren(...s.positions.map(positionCard));
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
  renderPositions(s);
  renderSignals(s);
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
    for (const id of ["kpis", "positions", "signals", "ladder", "closed", "chart"]) $(id).replaceChildren();
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
