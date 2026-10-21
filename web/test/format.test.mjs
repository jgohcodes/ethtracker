import test from "node:test";
import assert from "node:assert/strict";

import {
  usd, signedUsd, fixed, pct, signedPct, ageSeconds, freshness, humanAge, actionInfo, biasGlyph,
  priceLadder, isValidSymbol, FRESH_SECONDS, STALE_SECONDS, ACTIONS,
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
