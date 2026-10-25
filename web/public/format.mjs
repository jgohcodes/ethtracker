// Pure helpers for the dashboard: number formatting, snapshot freshness and the ordering of the price ladder.
// Kept free of the DOM so they can be tested under Node.

export const FRESH_SECONDS = 180; // for the 60-second default cadence
export const STALE_SECONDS = 600; // never call anything stale sooner than this

export const finite = (x) => typeof x === "number" && Number.isFinite(x);

function grouped(x, digits) {
  return Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function usd(x, digits = 2) {
  return finite(x) ? `${x < 0 ? "-" : ""}$${grouped(x, digits)}` : "n/a";
}

export function signedUsd(x, digits = 2) {
  return finite(x) ? `${x < 0 ? "-" : "+"}$${grouped(x, digits)}` : "n/a";
}

export function fixed(x, digits = 2) {
  return finite(x) ? x.toFixed(digits) : "n/a";
}

// Fractions in, percent out: pct(0.0123) is "1.2%".
export function pct(x, digits = 1) {
  return finite(x) ? `${(x * 100).toFixed(digits)}%` : "n/a";
}

// Odds: whole percents, but never a bare 0% or 100% for something merely unlikely or likely.
export function chance(x) {
  if (!finite(x)) return "n/a";
  if (x > 0 && x < 0.005) return "<1%";
  if (x < 1 && x > 0.995) return ">99%";
  return pct(x, 0);
}

export function eth(x, digits = 4) {
  return finite(x) ? `${x.toFixed(digits)} ETH` : "n/a";
}

// Funding is quoted per 8 hours and is tiny: four decimals of a percent.
export function funding8h(x) {
  return finite(x) ? `${(x * 100).toFixed(4)}% / 8h` : "n/a";
}

export function signedPct(x, digits = 1) {
  return finite(x) ? `${x < 0 ? "-" : "+"}${(Math.abs(x) * 100).toFixed(digits)}%` : "n/a";
}

export function ageSeconds(iso, now = Date.now()) {
  const then = Date.parse(iso);
  return Number.isNaN(then) ? null : Math.max(0, (now - then) / 1000);
}

// Judged against the cadence the publisher declares: current until one interval plus slack has passed,
// stale after two. A daily snapshot is current for 30 hours; a 60-second one goes stale after 10 minutes.
export function freshnessThresholds(interval = 60) {
  const step = finite(interval) && interval > 0 ? interval : 60;
  const grace = Math.max(120, 0.25 * step);
  return { fresh: step + grace, stale: Math.max(STALE_SECONDS, 2 * step + grace) };
}

export function freshness(age, interval = 60) {
  if (age === null || age === undefined) return "unknown";
  const limits = freshnessThresholds(interval);
  if (age <= limits.fresh) return "fresh";
  if (age <= limits.stale) return "delayed";
  return "stale";
}

export function describeInterval(seconds) {
  if (!finite(seconds) || seconds <= 0) return "at an unknown interval";
  if (seconds === 86400) return "daily";
  if (seconds % 86400 === 0) return `every ${seconds / 86400} days`;
  if (seconds >= 3600) return `every ${Math.round(seconds / 3600)}h`;
  if (seconds >= 60) return `every ${Math.round(seconds / 60)} min`;
  return `every ${Math.round(seconds)}s`;
}

export function humanAge(seconds) {
  if (seconds === null || seconds === undefined) return "unknown";
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

// Status tone and a distinct glyph per action, so meaning never rests on colour alone.
export const ACTIONS = {
  HOLD: { tone: "good", icon: "✓", label: "Hold" },
  TIGHTEN: { tone: "warning", icon: "↑", label: "Tighten stop" },
  TAKE_PARTIAL: { tone: "serious", icon: "◐", label: "Take partial" },
  EXIT: { tone: "critical", icon: "✕", label: "Exit" },
};

export function actionInfo(action) {
  return ACTIONS[action] ?? { tone: "warning", icon: "?", label: String(action) };
}

// The adding plan: a liquidation price at or under the floor is the plan working; above it needs action.
export const PLAN_STATUS = {
  "at the floor": { tone: "good", icon: "✓", label: "At the floor" },
  "below the floor": { tone: "good", icon: "✓", label: "Under the floor" },
  "above the floor": { tone: "critical", icon: "✕", label: "Above the floor" },
  "no position": { tone: "muted", icon: "•", label: "No position yet" },
};

// Read through the plan's risk: a downtrend and crowded longs make a fall to the floor likelier.
export const TREND = {
  Uptrend: { tone: "good", icon: "▲", label: "Uptrend" },
  Downtrend: { tone: "serious", icon: "▼", label: "Downtrend" },
  Mixed: { tone: "muted", icon: "•", label: "Mixed" },
};

export const CROWDING = {
  "Crowded longs": { tone: "warning", icon: "!", label: "Crowded longs" },
  Normal: { tone: "muted", icon: "•", label: "Normal" },
  "Light longs": { tone: "good", icon: "✓", label: "Light longs" },
};

// The signal score's labels, from -1 to +1: direction by glyph and position, never colour alone.
export const SCORE = {
  "Strongly positive": { tone: "good", icon: "▲", label: "Strongly positive" },
  "Leaning positive": { tone: "good", icon: "△", label: "Leaning positive" },
  Neutral: { tone: "muted", icon: "•", label: "Neutral" },
  "Leaning negative": { tone: "serious", icon: "▽", label: "Leaning negative" },
  "Strongly negative": { tone: "critical", icon: "▼", label: "Strongly negative" },
};

export function signedFixed(x, digits = 2) {
  return finite(x) ? `${x > 0 ? "+" : x < 0 ? "−" : "±"}${Math.abs(x).toFixed(digits)}` : "n/a";
}

export const stateWord = (s) => (s === null || s === undefined ? "n/a" : s > 0 ? "positive" : s < 0 ? "negative" : "neutral");

// "14 days ago positive, 7 days ago neutral, 3 days ago negative, now negative": a strip read aloud.
export function stripSummary(strip) {
  const n = strip.length;
  const at = (back) => (n - 1 - back >= 0 ? stateWord(strip[n - 1 - back]) : "n/a");
  return `14 days ago ${at(14)}, 7 days ago ${at(7)}, 3 days ago ${at(3)}, now ${at(0)}`;
}

const ordinal = (k) => {
  const tail = k % 100 >= 11 && k % 100 <= 13 ? "th" : { 1: "st", 2: "nd", 3: "rd" }[k % 10] ?? "th";
  return `${k}${tail}`;
};
const minus = (x) => (x < 0 ? "−" : "+");

// A signal's reading in its own unit: a return, std devs, RSI points, a percentile...
export function reading(x, unit) {
  if (!finite(x)) return "n/a";
  switch (unit) {
    case "pct": return `${minus(x)}${(Math.abs(x) * 100).toFixed(1)}%`;
    case "share": return `${minus(x)}${(Math.abs(x) * 100).toFixed(0)}%`;
    case "percentile": return `${ordinal(Math.round(x * 100))} percentile`;
    case "z": return `${minus(x)}${Math.abs(x).toFixed(1)} sd`;
    case "index": return x.toFixed(0);
    case "ratio": return `${x.toFixed(2)}x`;
    case "rate8h": return `${minus(x)}${(Math.abs(x) * 100).toFixed(4)}% / 8h`;
    default: return String(x);
  }
}

// "28-day return: 14 days ago +3.1%, 7 days ago −4.0%, 3 days ago −9.9%, now −12.6%; flips at +0.0%": a line read aloud.
export function lineSummary(line) {
  const v = line.values, n = v.length;
  const at = (back) => reading(n - 1 - back >= 0 ? v[n - 1 - back] : null, line.unit);
  const flips = (line.thresholds ?? []).map((t) => reading(t, line.unit)).join(" and ");
  return `${line.label}: 14 days ago ${at(14)}, 7 days ago ${at(7)}, 3 days ago ${at(3)}, now ${at(0)}` + (flips ? `; flips at ${flips}` : "");
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
// "2026-07-27" -> "27 Jul": an axis label, read from the string so no time zone can shift the day.
export function shortDate(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso ?? "");
  return m ? `${Number(m[3])} ${MONTHS[Number(m[2]) - 1]}` : "";
}

// A compact time for a table cell: "2026-10-25T17:02:11Z" -> "25 Oct 17:02".
export function shortTime(iso) {
  const m = /^\d{4}-\d{2}-\d{2}T(\d{2}:\d{2})/.exec(iso ?? "");
  return m ? `${shortDate(iso)} ${m[1]}` : "";
}

// An exchange time, "2026-10-25T17:02:11.123Z" -> "2026-10-25 17:02 UTC".
export function utcTime(iso) {
  const m = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/.exec(iso ?? "");
  return m ? `${m[1]} ${m[2]} UTC` : "";
}

// The score card's words for a score: the same edges as the Python side (each bucket includes its upper edge).
export const scoreLabel = (v) => (!finite(v) ? null : v <= -0.5 ? "Strongly negative" : v <= -0.2 ? "Leaning negative"
  : v <= 0.2 ? "Neutral" : v <= 0.5 ? "Leaning positive" : "Strongly positive");

export function lookup(table, key) {
  return table[key] ?? { tone: "muted", icon: "?", label: String(key) };
}

export const biasGlyph = (bias) => (bias > 0 ? "▲" : bias < 0 ? "▼" : "•");

export const MARKET_LEVELS = [
  ["sma_20", "20d average"], ["sma_50", "50d average"], ["sma_100", "100d average"], ["sma_200", "200d average"],
  ["avwap_low", "AVWAP from the low"], ["avwap_year", "AVWAP from the year open"],
  ["val", "Value-area low"], ["poc", "Point of control"], ["vah", "Value-area high"],
];

// One list, highest price first, mixing market levels, the mark and each position's own levels,
// so the order of everything relative to price reads at a glance (the embedded chart cannot draw them).
export function priceLadder(snap) {
  const rows = [{ kind: "mark", label: "Mark", price: snap.price }];
  for (const [key, label] of MARKET_LEVELS) {
    if (finite(snap.levels?.[key])) rows.push({ kind: "market", label, price: snap.levels[key] });
  }
  const plan = snap.plan;
  if (plan) {
    rows.push({ kind: "yours", label: "Plan floor", price: plan.floor });
    if (finite(plan.liq_price) && plan.liq_price > 0) rows.push({ kind: "yours", label: "Liquidation", price: plan.liq_price });
  }
  for (const p of snap.positions ?? []) {
    const tag = p.lots ? "Average" : `#${p.id} ${p.side}`;
    rows.push({ kind: "yours", label: `${tag} entry`, price: p.entry });
    if (finite(p.stop) && !(plan && p.stop === plan.floor)) rows.push({ kind: "yours", label: `${tag} stop`, price: p.stop });
    if (finite(p.target)) rows.push({ kind: "yours", label: `${tag} target`, price: p.target });
    if (finite(p.suggested_stop)) rows.push({ kind: "yours", label: `${tag} suggested stop`, price: p.suggested_stop });
  }
  return rows
    .filter((row) => finite(row.price))
    .sort((a, b) => b.price - a.price)
    .map((row) => ({ ...row, distance: snap.price ? row.price / snap.price - 1 : null }));
}

export function isValidSymbol(symbol) {
  return typeof symbol === "string" && /^[A-Za-z0-9_.:!-]{1,40}$/.test(symbol);
}
