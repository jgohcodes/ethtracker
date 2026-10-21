"use strict";

// The dashboard's two endpoints. They only store and return a snapshot that `ethtracker publish`
// computed on your machine, so this app needs no data libraries and holds no exchange keys.
//
//   POST /api/ingest    Authorization: Bearer $INGEST_TOKEN    (held by your machine only)
//   GET  /api/snapshot  Authorization: Bearer $DASHBOARD_TOKEN (held by the browser)
//
// Two tokens on purpose: the browser's read token cannot overwrite what the dashboard shows, and
// the write token never leaves your machine.

const crypto = require("node:crypto");

const SNAPSHOT_KEY = "ethtracker:snapshot";
const SUPPORTED_VERSION = 1;
const MAX_BYTES = 256 * 1024;
const TTL_SECONDS = 7 * 24 * 3600; // a snapshot nobody refreshed for a week disappears instead of lingering
const MAX_POSITIONS = 50;
const MAX_SIGNALS = 100;

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

class ConfigError extends Error {}

function bearer(req) {
  const match = /^Bearer\s+(\S+)\s*$/i.exec(req.headers?.authorization ?? "");
  return match ? match[1] : "";
}

// Hash both sides first so timingSafeEqual always compares equal-length buffers.
function tokenMatches(provided, expected) {
  if (!provided || !expected) return false;
  const digest = (value) => crypto.createHash("sha256").update(value).digest();
  return crypto.timingSafeEqual(digest(provided), digest(expected));
}

function send(res, status, body, headers = {}) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  for (const [key, value] of Object.entries(headers)) res.setHeader(key, value);
  res.end(JSON.stringify(body));
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    throw new HttpError(400, "body is not valid JSON");
  }
}

async function readBody(req) {
  if (req.body !== undefined && req.body !== null) {
    // Vercel pre-parses JSON bodies; accept an object, a string or a buffer.
    if (Buffer.isBuffer(req.body)) return parseJson(req.body.toString("utf8"));
    if (typeof req.body === "string") return parseJson(req.body);
    return req.body;
  }
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BYTES) throw new HttpError(413, "snapshot too large");
    chunks.push(chunk);
  }
  return parseJson(Buffer.concat(chunks).toString("utf8"));
}

// Returns the first problem with a snapshot, or null. Deliberately shallow: the page escapes everything
// it renders, so this only has to keep junk and oversized payloads out of storage.
function validate(snap) {
  if (snap === null || typeof snap !== "object" || Array.isArray(snap)) return "snapshot must be a JSON object";
  if (snap.version !== SUPPORTED_VERSION) return `unsupported snapshot version (expected ${SUPPORTED_VERSION})`;
  if (typeof snap.generated_at !== "string" || Number.isNaN(Date.parse(snap.generated_at))) {
    return "generated_at must be an ISO timestamp";
  }
  if (typeof snap.price !== "number" || !Number.isFinite(snap.price) || snap.price <= 0) {
    return "price must be a positive number";
  }
  if (!Array.isArray(snap.positions) || snap.positions.length > MAX_POSITIONS) {
    return `positions must be an array of at most ${MAX_POSITIONS}`;
  }
  if (!Array.isArray(snap.signals) || snap.signals.length > MAX_SIGNALS) {
    return `signals must be an array of at most ${MAX_SIGNALS}`;
  }
  return null;
}

let cachedClient;

// Upstash Redis through the Vercel Marketplace. The SDK accepts either naming scheme, and so do we.
function defaultRedis(env = process.env) {
  const url = env.UPSTASH_REDIS_REST_URL || env.KV_REST_API_URL;
  const token = env.UPSTASH_REDIS_REST_TOKEN || env.KV_REST_API_TOKEN;
  if (!url || !token) {
    throw new ConfigError(
      "Redis is not configured: connect an Upstash Redis store to this Vercel project " +
        "(it provides UPSTASH_REDIS_REST_URL or KV_REST_API_URL, and a token)",
    );
  }
  if (!cachedClient) {
    const { Redis } = require("@upstash/redis");
    cachedClient = new Redis({ url, token });
  }
  return cachedClient;
}

function createHandlers({ getRedis, env = process.env, now = () => new Date() }) {
  const storageFailure = (res, error) =>
    error instanceof ConfigError
      ? send(res, 500, { error: error.message })
      : send(res, 502, { error: "storage unavailable" }); // never echo the underlying error: it can name hosts

  async function snapshot(req, res) {
    if (req.method !== "GET") return send(res, 405, { error: "method not allowed" }, { Allow: "GET" });
    if (!env.DASHBOARD_TOKEN) return send(res, 500, { error: "DASHBOARD_TOKEN is not configured" });
    if (!tokenMatches(bearer(req), env.DASHBOARD_TOKEN)) return send(res, 401, { error: "unauthorized" });

    let raw;
    try {
      raw = await getRedis().get(SNAPSHOT_KEY);
    } catch (error) {
      return storageFailure(res, error);
    }
    if (raw === null || raw === undefined) return send(res, 404, { error: "no snapshot published yet" });
    if (typeof raw !== "string") return send(res, 200, raw); // the SDK already parsed the JSON
    try {
      return send(res, 200, JSON.parse(raw));
    } catch {
      return send(res, 502, { error: "stored snapshot is corrupt" });
    }
  }

  async function ingest(req, res) {
    if (req.method !== "POST") return send(res, 405, { error: "method not allowed" }, { Allow: "POST" });
    if (!env.INGEST_TOKEN) return send(res, 500, { error: "INGEST_TOKEN is not configured" });
    if (!tokenMatches(bearer(req), env.INGEST_TOKEN)) return send(res, 401, { error: "unauthorized" });

    let body;
    try {
      body = await readBody(req);
      if (Buffer.byteLength(JSON.stringify(body)) > MAX_BYTES) throw new HttpError(413, "snapshot too large");
    } catch (error) {
      return send(res, error.status ?? 400, { error: error.message });
    }
    const problem = validate(body);
    if (problem) return send(res, 400, { error: problem });

    const stored = { ...body, received_at: now().toISOString() };
    try {
      await getRedis().set(SNAPSHOT_KEY, JSON.stringify(stored), { ex: TTL_SECONDS });
    } catch (error) {
      return storageFailure(res, error);
    }
    return send(res, 200, { ok: true, received_at: stored.received_at });
  }

  return { snapshot, ingest };
}

module.exports = {
  createHandlers, defaultRedis, tokenMatches, validate, ConfigError,
  SNAPSHOT_KEY, MAX_BYTES, TTL_SECONDS, MAX_POSITIONS, MAX_SIGNALS, SUPPORTED_VERSION,
};
