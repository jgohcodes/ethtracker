// Pure helpers for the dashboard: number formatting, snapshot freshness and the ordering of the price ladder.
// Kept free of the DOM so they can be tested under Node.

export const FRESH_SECONDS = 180; // `ethtracker publish` runs every 60s by default
export const STALE_SECONDS = 600;

const finite = (x) => typeof x === "number" && Number.isFinite(x);

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

export function signedPct(x, digits = 1) {
  return finite(x) ? `${x < 0 ? "-" : "+"}${(Math.abs(x) * 100).toFixed(digits)}%` : "n/a";
}

export function ageSeconds(iso, now = Date.now()) {
  const then = Date.parse(iso);
  return Number.isNaN(then) ? null : Math.max(0, (now - then) / 1000);
}

export function freshness(age) {
  if (age === null || age === undefined) return "unknown";
  if (age <= FRESH_SECONDS) return "fresh";
  if (age <= STALE_SECONDS) return "delayed";
  return "stale";
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
  for (const p of snap.positions ?? []) {
    const tag = `#${p.id} ${p.side}`;
    rows.push({ kind: "yours", label: `${tag} entry`, price: p.entry });
    if (finite(p.stop)) rows.push({ kind: "yours", label: `${tag} stop`, price: p.stop });
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
