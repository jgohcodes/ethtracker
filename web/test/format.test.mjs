import test from "node:test";
import assert from "node:assert/strict";

import {
  usd, signedUsd, fixed, pct, signedPct, ageSeconds, freshness, humanAge, actionInfo, biasGlyph,
  priceLadder, isValidSymbol, FRESH_SECONDS, STALE_SECONDS, ACTIONS, freshnessThresholds, describeInterval,
  chance, eth, funding8h, SCORE, signedFixed, stateWord, stripSummary, PLAN_STATUS, TREND, CROWDING, lookup, reading, lineSummary,
  shortDate, utcTime, scoreLabel,
} from "../public/format.mjs";

test("money formatting groups thousands, signs deltas and survives missing values", () => {
  assert.equal(usd(1234567.891), "$1,234,567.89");
  assert.equal(usd(-0.5), "-$0.50");
  assert.equal(usd(3000, 0), "$3,000");
  assert.equal(signedUsd(12.3), "+$12.30");
  assert.equal(signedUsd(-12.3), "-$12.30");
  for (const bad of [null, undefined, NaN, Infinity, "3", {}]) {
    assert.equal(usd(bad), "n/a");
    assert.equal(signedUsd(bad), "n/a");
    assert.equal(pct(bad), "n/a");
    assert.equal(signedPct(bad), "n/a");
  }
});

test("fixed formats numbers and survives missing values", () => {
  assert.equal(fixed(2.4814, 2), "2.48");
  assert.equal(fixed(97.60859, 4), "97.6086");
  assert.equal(fixed(null), "n/a");
  assert.equal(fixed(Infinity), "n/a");
});

test("percentages take fractions", () => {
  assert.equal(pct(0.0123), "1.2%");
  assert.equal(pct(1), "100.0%");
  assert.equal(pct(0.5, 0), "50%");
  assert.equal(signedPct(0.1234), "+12.3%");
  assert.equal(signedPct(-0.0456), "-4.6%");
  assert.equal(signedPct(0), "+0.0%");
});

test("freshness scales with the declared cadence", () => {
  const hour = 3600;
  assert.equal(freshness(20 * hour, 86400), "fresh");
  assert.equal(freshness(29 * hour, 86400), "fresh");
  assert.equal(freshness(31 * hour, 86400), "delayed");
  assert.equal(freshness(53 * hour, 86400), "delayed");
  assert.equal(freshness(55 * hour, 86400), "stale");
  assert.deepEqual(freshnessThresholds(86400), { fresh: 108000, stale: 194400 });
  assert.deepEqual(freshnessThresholds(60), { fresh: FRESH_SECONDS, stale: STALE_SECONDS });
  for (const junk of [0, -5, null, undefined, NaN]) assert.deepEqual(freshnessThresholds(junk), freshnessThresholds(60));
});

test("describeInterval reads naturally", () => {
  assert.equal(describeInterval(86400), "daily");
  assert.equal(describeInterval(172800), "every 2 days");
  assert.equal(describeInterval(14400), "every 4h");
  assert.equal(describeInterval(60), "every 1 min");
  assert.equal(describeInterval(30), "every 30s");
  assert.equal(describeInterval(null), "at an unknown interval");
});

test("freshness judges the age of the last snapshot", () => {
  assert.equal(freshness(0), "fresh");
  assert.equal(freshness(FRESH_SECONDS), "fresh");
  assert.equal(freshness(FRESH_SECONDS + 1), "delayed");
  assert.equal(freshness(STALE_SECONDS), "delayed");
  assert.equal(freshness(STALE_SECONDS + 1), "stale");
  assert.equal(freshness(null), "unknown");
  assert.equal(freshness(undefined), "unknown");
});

test("ageSeconds never goes negative and rejects junk", () => {
  const now = Date.parse("2026-10-21T12:00:30Z");
  assert.equal(ageSeconds("2026-10-21T12:00:00Z", now), 30);
  assert.equal(ageSeconds("2026-10-21T12:05:00Z", now), 0); // a receive time slightly ahead of this clock
  assert.equal(ageSeconds("not a date", now), null);
});

test("humanAge picks a sensible unit", () => {
  assert.equal(humanAge(5.9), "5s ago");
  assert.equal(humanAge(125), "2m ago");
  assert.equal(humanAge(7200), "2h ago");
  assert.equal(humanAge(200000), "2d ago");
  assert.equal(humanAge(null), "unknown");
});

test("every action has a tone, a label and its own glyph", () => {
  const icons = Object.values(ACTIONS).map((a) => a.icon);
  assert.equal(new Set(icons).size, icons.length, "glyphs must differ so colour is never the only cue");
  assert.deepEqual(Object.keys(ACTIONS), ["HOLD", "TIGHTEN", "TAKE_PARTIAL", "EXIT"]);
  assert.equal(actionInfo("EXIT").tone, "critical");
  assert.equal(actionInfo("HOLD").tone, "good");
  assert.deepEqual(actionInfo("SOMETHING_NEW"), { tone: "warning", icon: "?", label: "SOMETHING_NEW" });
});

test("bias glyphs differ by shape", () => {
  assert.deepEqual([biasGlyph(1), biasGlyph(-1), biasGlyph(0)], ["▲", "▼", "•"]);
});

const SNAP = {
  price: 3000,
  levels: { sma_20: 2900, sma_200: 3300, poc: 2700, avwap_low: NaN, bogus: 1 },
  positions: [
    { id: 1, side: "long", entry: 2800, stop: 2688, target: 3500, suggested_stop: 2950 },
    { id: 2, side: "short", entry: 3100, stop: 3200, target: null, suggested_stop: null },
  ],
};

test("the price ladder sorts everything by price, highest first, with distances from the mark", () => {
  const ladder = priceLadder(SNAP);
  const prices = ladder.map((row) => row.price);
  assert.deepEqual(prices, [...prices].sort((a, b) => b - a));
  const mark = ladder.find((row) => row.kind === "mark");
  assert.equal(mark.distance, 0);
  const sma200 = ladder.find((row) => row.label === "200d average");
  assert.ok(Math.abs(sma200.distance - 0.1) < 1e-9);
  assert.equal(ladder.find((row) => row.label === "#1 long suggested stop").price, 2950);
  assert.equal(ladder.find((row) => row.label === "#1 long target").kind, "yours");
});

test("the ladder skips levels that are missing or not finite and ignores unknown keys", () => {
  const labels = priceLadder(SNAP).map((row) => row.label);
  assert.ok(!labels.includes("AVWAP from the low"));
  assert.ok(!labels.includes("#2 short target"));
  assert.ok(!labels.includes("#2 short suggested stop"));
  assert.ok(!labels.some((label) => /bogus/i.test(label)));
  assert.deepEqual(priceLadder({ price: 100 }).map((row) => row.label), ["Mark"]);
});

test("symbol overrides are limited to characters TradingView symbols use", () => {
  assert.equal(isValidSymbol("COINBASE:ETHUSD"), true);
  assert.equal(isValidSymbol("DERIBIT:ETHUSDC.P"), true);
  for (const bad of ["", "A".repeat(41), 'x"};alert(1)//', "a b", "<script>", null, undefined, 5]) {
    assert.equal(isValidSymbol(bad), false, String(bad));
  }
});

test("score labels differ by glyph and the strip reads aloud at 14, 7 and 3 days and now", () => {
  const icons = Object.values(SCORE).map((s) => s.icon);
  assert.equal(new Set(icons).size, icons.length);
  assert.deepEqual([signedFixed(0.3), signedFixed(-0.25), signedFixed(0), signedFixed(null)], ["+0.30", "−0.25", "±0.00", "n/a"]);
  assert.deepEqual([stateWord(1), stateWord(0), stateWord(-1), stateWord(null)], ["positive", "neutral", "negative", "n/a"]);
  const strip = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1];
  assert.equal(stripSummary(strip), "14 days ago positive, 7 days ago neutral, 3 days ago negative, now negative");
});

test("odds never round a small chance to 0% or a near-certainty to 100%", () => {
  assert.deepEqual([chance(0.000134), chance(0), chance(0.273), chance(0.999), chance(1), chance(null)],
    ["<1%", "0%", "27%", ">99%", "100%", "n/a"]);
});

test("ETH sizes and funding rates format and survive missing values", () => {
  assert.equal(eth(4.57281), "4.5728 ETH");
  assert.equal(eth(null), "n/a");
  assert.equal(funding8h(0.0000123), "0.0012% / 8h");
  assert.equal(funding8h(undefined), "n/a");
});

test("plan, trend and crowding states each carry a tone, a label and a distinct glyph within their set", () => {
  for (const table of [TREND, CROWDING]) {
    const icons = Object.values(table).map((s) => s.icon);
    assert.equal(new Set(icons).size, icons.length);
  }
  assert.equal(PLAN_STATUS["above the floor"].tone, "critical");
  assert.equal(PLAN_STATUS["at the floor"].tone, "good");
  assert.deepEqual(Object.keys(PLAN_STATUS).sort(), ["above the floor", "at the floor", "below the floor", "no position"]);
  assert.deepEqual(lookup(TREND, "Sideways"), { tone: "muted", icon: "?", label: "Sideways" });
});

test("under a plan the ladder shows the floor and the liquidation once, not each buy's stop", () => {
  const snap = {
    price: 2336,
    levels: {},
    plan: { floor: 1250, liq_price: 1248.8 },
    positions: [{ id: 1, side: "long", entry: 2300, stop: 1250, target: null, suggested_stop: null, lots: [{}, {}] }],
  };
  const labels = priceLadder(snap).map((row) => row.label);
  assert.deepEqual(labels, ["Mark", "Average entry", "Plan floor", "Liquidation"]);
});

test("a signal's reading prints in its own unit and its line reads aloud with where it flips", () => {
  assert.deepEqual(
    [reading(-0.1264, "pct"), reading(-0.3333, "share"), reading(0.30, "percentile"), reading(0.91, "percentile"),
      reading(0.12, "percentile"), reading(-1.26, "z"), reading(36.4, "index"), reading(0.97, "ratio"), reading(0.0000071, "rate8h"), reading(null, "pct")],
    ["−12.6%", "−33%", "30th percentile", "91st percentile", "12th percentile", "−1.3 sd", "36", "0.97x", "+0.0007% / 8h", "n/a"]);
  const values = Array.from({ length: 30 }, (_, i) => (i - 20) / 100);
  assert.equal(lineSummary({ label: "28-day return", unit: "pct", values, thresholds: [0] }),
    "28-day return: 14 days ago −5.0%, 7 days ago +2.0%, 3 days ago +6.0%, now +9.0%; flips at +0.0%");
});

test("axis dates, exchange times and score words", () => {
  assert.deepEqual([shortDate("2026-07-27"), shortDate("2026-01-01T00:00:00Z"), shortDate(null)], ["27 Jul", "1 Jan", ""]);
  assert.deepEqual([utcTime("2026-10-25T17:02:11.123456Z"), utcTime("bad")], ["2026-10-25 17:02 UTC", ""]);
  assert.deepEqual([-0.54, -0.5, -0.2, 0, 0.2, 0.5, 0.51, null].map(scoreLabel),
    ["Strongly negative", "Strongly negative", "Leaning negative", "Neutral", "Neutral", "Leaning positive", "Strongly positive", null]);
});
