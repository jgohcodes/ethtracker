"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");

const { createHandlers, STATE_LIMITS } = require("../lib/handlers");

const ENV = { DASHBOARD_TOKEN: "view-token-123", INGEST_TOKEN: "ingest-token-456" };
const NOW = new Date("2026-10-24T00:20:00.000Z");
const JOURNAL = JSON.stringify({ equity: 100000, peak_equity: 100000, next_id: 2, positions: [{ id: 1, side: "long" }] });

function fakeRedis(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    store, sets: [],
    async get(key) { return store.has(key) ? store.get(key) : null; },
    async set(key, value, options) { this.sets.push({ key, value, options }); store.set(key, value); return "OK"; },
  };
}

async function withServer({ redis = fakeRedis(), env = ENV } = {}, fn) {
  const handlers = createHandlers({ getRedis: () => redis, env, now: () => NOW });
  const server = http.createServer((req, res) => {
    if (new URL(req.url, "http://x").pathname === "/api/state") return handlers.state(req, res);
    res.statusCode = 404;
    res.end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}/api/state`;
  const call = async (query, { method = "GET", token = ENV.INGEST_TOKEN, body } = {}) => {
    const response = await fetch(`${base}${query}`, {
      method,
      headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(body !== undefined ? { "Content-Type": "application/json" } : {}) },
      body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
    });
    const text = await response.text();
    return { status: response.status, headers: response.headers, text, json: () => JSON.parse(text) };
  };
  try {
    await fn({ call, redis });
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
}

test("state: only GET and PUT, and only with the write token", () => withServer({}, async ({ call }) => {
  const post = await call("?name=journal", { method: "POST", body: { content: JOURNAL } });
  assert.equal(post.status, 405);
  assert.equal(post.headers.get("allow"), "GET, PUT");
  assert.equal((await call("?name=journal", { token: null })).status, 401); // null, not undefined: undefined would fall back to the default token
  assert.equal((await call("?name=journal", { token: ENV.DASHBOARD_TOKEN })).status, 401, "the browser's token must not read the raw journal");
  assert.equal((await call("?name=journal", { method: "PUT", token: ENV.DASHBOARD_TOKEN, body: { content: JOURNAL } })).status, 401);
}));

test("state: refuses everything when no write token is configured", () =>
  withServer({ env: { DASHBOARD_TOKEN: "v" } }, async ({ call }) => {
    const response = await call("?name=journal", { token: "anything" });
    assert.equal(response.status, 500);
    assert.match(response.json().error, /INGEST_TOKEN is not configured/);
  }));

test("state: names come from an allowlist", () => withServer({}, async ({ call }) => {
  for (const query of ["", "?name=", "?name=snapshot", "?name=__proto__", "?name=journal/../x", "?name=constructor"]) {
    assert.equal((await call(query)).status, 400, query);
  }
  assert.deepEqual(Object.keys(STATE_LIMITS), ["journal", "derivs"]);
}));

test("state: 404 until stored, then a round trip with a server timestamp and no expiry", () =>
  withServer({}, async ({ call, redis }) => {
    assert.equal((await call("?name=journal")).status, 404);
    const put = await call("?name=journal", { method: "PUT", body: { content: JOURNAL } });
    assert.equal(put.status, 200);
    assert.deepEqual(put.json(), { ok: true, updated_at: NOW.toISOString() });
    assert.equal(redis.sets[0].key, "ethtracker:state:journal");
    assert.equal(redis.sets[0].options, undefined, "the journal must not expire");
    assert.deepEqual((await call("?name=journal")).json(), { content: JOURNAL, updated_at: NOW.toISOString() });
    assert.equal((await call("?name=derivs")).status, 404, "names are stored separately");
  }));

test("state: a value the SDK already parsed is returned as is", () =>
  withServer({ redis: fakeRedis({ "ethtracker:state:derivs": { content: "time,oi\n", updated_at: "x" } }) }, async ({ call }) => {
    assert.deepEqual((await call("?name=derivs")).json(), { content: "time,oi\n", updated_at: "x" });
  }));

test("state: the body must carry string content, and a journal must be a JSON object with positions", () =>
  withServer({}, async ({ call, redis }) => {
    const put = (name, body) => call(`?name=${name}`, { method: "PUT", body });
    assert.equal((await put("journal", "{broken")).status, 400);
    assert.equal((await put("journal", { content: 5 })).status, 400);
    assert.equal((await put("journal", { other: JOURNAL })).status, 400);
    assert.equal((await put("journal", { content: "not json" })).status, 400);
    assert.equal((await put("journal", { content: "[1,2]" })).status, 400);
    assert.equal((await put("journal", { content: JSON.stringify({ equity: 1 }) })).status, 400);
    assert.equal(redis.sets.length, 0);
    assert.equal((await put("derivs", { content: "time,open_interest\n2026-10-24,1\n" })).status, 200);
  }));

test("state: each name has its own size cap", () => withServer({}, async ({ call }) => {
  const rows = (bytes) => "time,oi\n" + "x".repeat(bytes);
  assert.equal((await call("?name=derivs", { method: "PUT", body: { content: rows(500 * 1024) } })).status, 200);
  assert.equal((await call("?name=derivs", { method: "PUT", body: { content: rows(STATE_LIMITS.derivs + 1) } })).status, 413);
  const bigJournal = JSON.stringify({ positions: [], pad: "x".repeat(STATE_LIMITS.journal) });
  assert.equal((await call("?name=journal", { method: "PUT", body: { content: bigJournal } })).status, 413);
}));

test("state: storage failures are 502 without the underlying error", () => {
  const broken = { async get() { throw new Error("secret-host.upstash.io"); }, async set() { throw new Error("secret-host.upstash.io"); } };
  return withServer({ redis: broken }, async ({ call }) => {
    for (const response of [await call("?name=journal"), await call("?name=journal", { method: "PUT", body: { content: JOURNAL } })]) {
      assert.equal(response.status, 502);
      assert.doesNotMatch(response.text, /secret-host/);
    }
  });
});

test("state: a corrupt stored document is a 502, not a crash", () =>
  withServer({ redis: fakeRedis({ "ethtracker:state:journal": "{nope" }) }, async ({ call }) => {
    assert.equal((await call("?name=journal")).status, 502);
  }));
