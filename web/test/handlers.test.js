"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");

const {
  createHandlers, tokenMatches, validate, defaultRedis, ConfigError,
  SNAPSHOT_KEY, MAX_BYTES, TTL_SECONDS, MAX_POSITIONS,
} = require("../lib/handlers");

const ENV = { DASHBOARD_TOKEN: "view-token-123", INGEST_TOKEN: "ingest-token-456" };
const GOOD = { version: 1, generated_at: "2026-10-21T12:00:00+00:00", price: 3000.5, positions: [], signals: [] };
const NOW = new Date("2026-10-21T12:00:05.000Z");

function fakeRedis(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    store, sets: [],
    async get(key) { return store.has(key) ? store.get(key) : null; },
    async set(key, value, options) { this.sets.push({ key, value, options }); store.set(key, value); return "OK"; },
  };
}

async function serve({ redis = fakeRedis(), env = ENV } = {}) {
  const handlers = createHandlers({ getRedis: () => redis, env, now: () => NOW });
  const server = http.createServer((req, res) => {
    const path = new URL(req.url, "http://x").pathname;
    if (path === "/api/snapshot") return handlers.snapshot(req, res);
    if (path === "/api/ingest") return handlers.ingest(req, res);
    res.statusCode = 404;
    res.end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  const call = async (path, { method = "GET", token, body, headers = {} } = {}) => {
    const response = await fetch(url + path, {
      method,
      headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...(body ? { "Content-Type": "application/json" } : {}), ...headers },
      body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
    });
    const text = await response.text();
    return { status: response.status, headers: response.headers, text, json: () => JSON.parse(text) };
  };
  return { call, redis, close: () => new Promise((resolve) => server.close(resolve)) };
}

async function withServer(options, fn) {
  const server = await serve(options);
  try {
    await fn(server);
  } finally {
    await server.close();
  }
}

test("tokenMatches compares exactly, whatever the lengths", () => {
  assert.equal(tokenMatches("abc", "abc"), true);
  assert.equal(tokenMatches("abc", "abd"), false);
  assert.equal(tokenMatches("abc", "abcd"), false);
  assert.equal(tokenMatches("abcd", "abc"), false);
  assert.equal(tokenMatches("", "abc"), false);
  assert.equal(tokenMatches("abc", ""), false);
  assert.equal(tokenMatches(undefined, "abc"), false);
});

test("snapshot: only GET is allowed", () => withServer({}, async ({ call }) => {
  const response = await call("/api/snapshot", { method: "POST", token: ENV.DASHBOARD_TOKEN, body: {} });
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("allow"), "GET");
}));

test("snapshot: refuses everything when no token is configured, rather than allowing everything", () =>
  withServer({ env: {} }, async ({ call }) => {
    const response = await call("/api/snapshot", { token: "anything" });
    assert.equal(response.status, 500);
    assert.match(response.json().error, /DASHBOARD_TOKEN is not configured/);
  }));

test("snapshot: rejects missing, wrong, prefix, query-string and non-bearer credentials", () =>
  withServer({ redis: fakeRedis({ [SNAPSHOT_KEY]: JSON.stringify(GOOD) }) }, async ({ call }) => {
    for (const attempt of [
      {},
      { token: "wrong" },
      { token: ENV.DASHBOARD_TOKEN.slice(0, -1) },
      { token: ENV.DASHBOARD_TOKEN + "x" },
      { token: ENV.INGEST_TOKEN },
      { headers: { Authorization: `Basic ${ENV.DASHBOARD_TOKEN}` } },
    ]) {
      assert.equal((await call("/api/snapshot", attempt)).status, 401, JSON.stringify(attempt));
    }
    assert.equal((await call(`/api/snapshot?token=${ENV.DASHBOARD_TOKEN}`)).status, 401);
    assert.equal((await call("/api/snapshot", { headers: { Authorization: `bearer ${ENV.DASHBOARD_TOKEN}` } })).status, 200);
  }));

test("snapshot: 404 until something has been published", () => withServer({}, async ({ call }) => {
  const response = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
  assert.equal(response.status, 404);
  assert.match(response.json().error, /no snapshot published/);
}));

test("snapshot: returns the stored snapshot and forbids caching", () =>
  withServer({ redis: fakeRedis({ [SNAPSHOT_KEY]: JSON.stringify({ ...GOOD, received_at: "r" }) }) }, async ({ call }) => {
    const response = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.equal(response.status, 200);
    assert.deepEqual(response.json(), { ...GOOD, received_at: "r" });
    assert.equal(response.headers.get("cache-control"), "no-store");
    assert.match(response.headers.get("content-type"), /application\/json/);
  }));

test("snapshot: accepts a value the SDK already parsed into an object", () =>
  withServer({ redis: fakeRedis({ [SNAPSHOT_KEY]: GOOD }) }, async ({ call }) => {
    const response = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.deepEqual(response.json(), GOOD);
  }));

test("snapshot: a corrupt stored value is a 502, not a crash", () =>
  withServer({ redis: fakeRedis({ [SNAPSHOT_KEY]: "{not json" }) }, async ({ call }) => {
    const response = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.equal(response.status, 502);
  }));

test("storage failures are 502 and never echo the underlying error", () => {
  const broken = { async get() { throw new Error("connect ECONNREFUSED secret-host.upstash.io:443"); }, async set() { throw new Error("secret-host.upstash.io"); } };
  return withServer({ redis: broken }, async ({ call }) => {
    const read = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    const write = await call("/api/ingest", { method: "POST", token: ENV.INGEST_TOKEN, body: GOOD });
    for (const response of [read, write]) {
      assert.equal(response.status, 502);
      assert.doesNotMatch(response.text, /secret-host|ECONNREFUSED/);
    }
  });
});

test("a missing Redis integration is a 500 that says how to fix it", () => {
  const unconfigured = { get() { throw new ConfigError("Redis is not configured: connect an Upstash Redis store"); } };
  return withServer({ redis: unconfigured }, async ({ call }) => {
    const response = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.equal(response.status, 500);
    assert.match(response.json().error, /connect an Upstash Redis store/);
  });
});

test("defaultRedis names the problem when the environment has no Redis credentials", () => {
  assert.throws(() => defaultRedis({}), (error) => error instanceof ConfigError && /UPSTASH_REDIS_REST_URL or KV_REST_API_URL/.test(error.message));
  assert.throws(() => defaultRedis({ KV_REST_API_URL: "https://x.upstash.io" }), ConfigError); // a URL without a token
});

test("ingest: only POST is allowed and a token must be configured", async () => {
  await withServer({}, async ({ call }) => {
    const response = await call("/api/ingest", { token: ENV.INGEST_TOKEN });
    assert.equal(response.status, 405);
    assert.equal(response.headers.get("allow"), "POST");
  });
  await withServer({ env: { DASHBOARD_TOKEN: "v" } }, async ({ call }) => {
    const response = await call("/api/ingest", { method: "POST", token: "x", body: GOOD });
    assert.equal(response.status, 500);
    assert.match(response.json().error, /INGEST_TOKEN is not configured/);
  });
});

test("ingest: the read token cannot write and the write token cannot read", () =>
  withServer({}, async ({ call, redis }) => {
    assert.equal((await call("/api/ingest", { method: "POST", token: ENV.DASHBOARD_TOKEN, body: GOOD })).status, 401);
    assert.equal((await call("/api/ingest", { method: "POST", body: GOOD })).status, 401);
    assert.equal((await call("/api/snapshot", { token: ENV.INGEST_TOKEN })).status, 401);
    assert.equal(redis.sets.length, 0);
  }));

test("ingest: stores the snapshot with a server receive time and an expiry, and the page can read it back", () =>
  withServer({}, async ({ call, redis }) => {
    const response = await call("/api/ingest", { method: "POST", token: ENV.INGEST_TOKEN, body: GOOD });
    assert.equal(response.status, 200);
    assert.deepEqual(response.json(), { ok: true, received_at: NOW.toISOString() });
    assert.equal(redis.sets.length, 1);
    assert.equal(redis.sets[0].key, SNAPSHOT_KEY);
    assert.deepEqual(redis.sets[0].options, { ex: TTL_SECONDS });
    const read = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.deepEqual(read.json(), { ...GOOD, received_at: NOW.toISOString() });
  }));

test("ingest: the client cannot forge the receive time that staleness is judged on", () =>
  withServer({}, async ({ call }) => {
    await call("/api/ingest", { method: "POST", token: ENV.INGEST_TOKEN, body: { ...GOOD, received_at: "2099-01-01T00:00:00Z" } });
    const read = await call("/api/snapshot", { token: ENV.DASHBOARD_TOKEN });
    assert.equal(read.json().received_at, NOW.toISOString());
  }));

test("ingest: rejects bad JSON, non-objects and invalid snapshots without storing them", () =>
  withServer({}, async ({ call, redis }) => {
    const post = (body) => call("/api/ingest", { method: "POST", token: ENV.INGEST_TOKEN, body });
    assert.equal((await post("{broken")).status, 400);
    assert.equal((await post([1, 2])).status, 400);
    assert.equal((await post({ ...GOOD, version: 2 })).status, 400);
    assert.equal((await post({ ...GOOD, generated_at: "yesterday-ish" })).status, 400);
    assert.equal((await post({ ...GOOD, price: -1 })).status, 400);
    assert.equal((await post({ ...GOOD, price: "3000" })).status, 400);
    assert.equal((await post({ ...GOOD, positions: "none" })).status, 400);
    assert.equal((await post({ ...GOOD, positions: new Array(MAX_POSITIONS + 1).fill({}) })).status, 400);
    assert.equal((await post({ ...GOOD, signals: undefined })).status, 400);
    assert.equal(redis.sets.length, 0);
  }));

test("ingest: refuses oversized payloads", () =>
  withServer({}, async ({ call, redis }) => {
    const huge = { ...GOOD, notes: ["x".repeat(1024)].concat(new Array(MAX_BYTES / 1024).fill("y".repeat(1024))) };
    const response = await call("/api/ingest", { method: "POST", token: ENV.INGEST_TOKEN, body: huge });
    assert.equal(response.status, 413);
    assert.equal(redis.sets.length, 0);
  }));

test("validate reports the first problem and null for a good snapshot", () => {
  assert.equal(validate(GOOD), null);
  assert.match(validate(null), /JSON object/);
  assert.match(validate({ ...GOOD, version: 0 }), /version/);
});
