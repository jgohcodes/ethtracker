"""Read-only access to your Coinbase account through a CDP API key with view permission only.

The key comes from the environment (COINBASE_API_KEY_JSON: the downloaded key file's contents, pasted whole into
a GitHub secret or exported on your machine). It is never read from a file in the repository and never printed.
Each request is signed with a two-minute JWT: ES256 for an EC key (a PEM private key), EdDSA for an Ed25519 key
(a base64 secret), with the key's id as subject and the request's method, host and path as its uri.

Two outputs, kept apart on purpose because this repository's logs are public:
  probe()    which endpoints answered, their status, field names, and yes/no for an ETH perpetual position or
             fill. No amounts, identifiers, counts or other products.
  account()  your ETH perpetual position, collateral and recent fills, for the private dashboard only.

Coinbase's international perpetuals moved to a Deribit-powered venue in October 2026; whether this API still
returns them is what the probe finds out. The parsing accepts amounts as {"value": "..."} objects or plain strings,
since the exact payloads were not available when this was written.
"""

from __future__ import annotations

import base64
import json
import os
import re
import secrets
import time
from dataclasses import dataclass

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

from .data import DataError

ENV = "COINBASE_API_KEY_JSON"
ENV_ID = "COINBASE_API_KEY_ID"  # the key's ID, when COINBASE_API_KEY_JSON holds only its secret
HOST = "api.coinbase.com"
BASE = f"https://{HOST}"
ETH_PERP = re.compile(r"^ETH[-_]")  # ETH-PERP-INTX, ETH-PERP, ETH_USDC-PERPETUAL ...
UUIDISH = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}|[0-9a-fA-F]{24,}")


@dataclass(frozen=True)
class Key:
    id: str
    private: object
    alg: str  # ES256 or EdDSA


PEM = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.+?-----END [A-Z ]*PRIVATE KEY-----", re.S)
KEY_NAME = re.compile(r"organizations/[\w-]+/apiKeys/[\w-]+")


def _bare_secret(text: str) -> bool:
    """An Ed25519 key's secret on its own: base64 of 32 or 64 bytes (44 or 88 characters)."""
    try:
        return len(base64.b64decode(text, validate=True)) in (32, 64)
    except ValueError:
        return False


def _parse(text: str, key_id: str | None = None) -> dict:
    """The key file's fields. Accepts the file as downloaded, the same with real line breaks inside the private key
    (as copied from a screen), or just the key name and the PEM block pasted one after the other."""
    text = text.strip().lstrip("\ufeff")
    for strict in (True, False):  # strict=False allows the raw line breaks a copied key carries
        try:
            data = json.loads(text, strict=strict)
            if isinstance(data, dict):
                return data
        except ValueError:
            pass
    pem, name = PEM.search(text), KEY_NAME.search(text)
    if pem and (name or key_id):
        return {"name": name.group(0) if name else key_id.strip(), "privateKey": pem.group(0)}
    if _bare_secret(text):
        if key_id:
            return {"id": key_id.strip(), "privateKey": text}
        raise DataError(
            f"{ENV} holds the key's secret on its own ({len(text)} characters of base64), without the key's ID. "
            f"Add the ID as a second secret named {ENV_ID} (the 'API key ID' shown for this key in the Coinbase "
            "Developer Platform), or replace this one with the whole downloaded .json file's text")
    raise DataError(
        f"{ENV} is not a key file: {len(text)} characters, starts with '{{': {'yes' if text.startswith('{') else 'no'}, "
        f"a private key block: {'yes' if pem else 'no'}, a key name (organizations/.../apiKeys/...): "
        f"{'yes' if name else 'no'}. Paste the whole downloaded .json file's text")


def load_key(text: str, key_id: str | None = None) -> Key:
    data = _parse(text, key_id)
    key_id = data.get("name") or data.get("id") or data.get("keyName")
    secret = data.get("privateKey") or data.get("private_key") or data.get("secret")
    if not key_id or not secret:
        raise DataError(f"{ENV} has no key id and private key (fields found: {', '.join(sorted(data))})")
    if "BEGIN" in secret:
        private = serialization.load_pem_private_key(secret.replace("\\n", "\n").encode(), password=None)
        if not isinstance(private, ec.EllipticCurvePrivateKey):
            raise DataError("the PEM key is not an EC key")
        return Key(key_id, private, "ES256")
    raw = base64.b64decode(secret)
    if len(raw) not in (32, 64):
        raise DataError("the Ed25519 secret should decode to 32 or 64 bytes")
    return Key(key_id, ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32]), "EdDSA")


def env_key() -> Key | None:
    text = os.environ.get(ENV)
    return load_key(text, os.environ.get(ENV_ID)) if text else None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def token(key: Key, method: str, path: str, style: str = "uri", now: int | None = None) -> str:
    """A signed JWT for one request. ``style`` "uri" is the App API's documented claim; "uris" is the CDP SDK's."""
    now = int(time.time()) if now is None else now
    uri = f"{method} {HOST}{path.split('?')[0]}"
    header = {"alg": key.alg, "kid": key.id, "typ": "JWT", "nonce": secrets.token_hex(16)}
    claims = {"sub": key.id, "iss": "cdp", "nbf": now, "exp": now + 120}
    claims.update({"uri": uri} if style == "uri" else {"uris": [uri]})
    signing = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(claims, separators=(',', ':')).encode())}"
    if key.alg == "ES256":
        r, s = decode_dss_signature(key.private.sign(signing.encode(), ec.ECDSA(hashes.SHA256())))
        signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    else:
        signature = key.private.sign(signing.encode())
    return f"{signing}.{_b64(signature)}"


class Client:
    def __init__(self, key: Key):
        self.key = key
        self.style = "uri"

    def get(self, path: str, params: dict | None = None) -> tuple[int, object]:
        """(status, parsed body or None). A 401 is retried once with the other claim style; the one that works stays."""
        for attempt in range(2):
            try:
                r = requests.get(BASE + path, params=params, timeout=20,
                                 headers={"Authorization": f"Bearer {token(self.key, 'GET', path, self.style)}"})
            except requests.RequestException as exc:
                raise DataError(f"could not reach {HOST}: {exc.__class__.__name__}") from exc
            if r.status_code == 401 and attempt == 0:
                self.style = "uris" if self.style == "uri" else "uri"
                continue
            try:
                body = r.json()
            except ValueError:
                body = None
            return r.status_code, body
        return r.status_code, None


# ---- what the endpoints hold, as shapes only (safe for public logs) -----------------------------------

def _shape(body) -> str:
    if isinstance(body, dict):
        return ", ".join(f"{k}[]" if isinstance(v, list) else k for k, v in body.items()) or "{}"
    return "a list" if isinstance(body, list) else "no JSON"


def _scrub(text: str) -> str:
    return UUIDISH.sub("<id>", text)[:120]


def probe(client: Client) -> list[str]:
    """One line per endpoint: status and shape. Product names and booleans only, never amounts or identifiers."""
    lines = []
    status, body = client.get("/api/v3/brokerage/key_permissions")
    if status == 200 and isinstance(body, dict):
        lines.append(f"key permissions: view {body.get('can_view')}, trade {body.get('can_trade')}, "
                     f"transfer {body.get('can_transfer')}, portfolio type {body.get('portfolio_type')}")
        if body.get("can_trade") or body.get("can_transfer"):
            lines.append("WARNING: this key can trade or transfer. Replace it with a view-only key.")
    else:
        lines.append(f"key permissions: HTTP {status} ({_scrub(str((body or {}).get('message', '')))})"
                     if isinstance(body, dict) else f"key permissions: HTTP {status}")
    lines.append(f"signature claim style that worked: {client.style}")
    for path, label in _endpoints(client):
        status, body = client.get(path, label[1])
        why = ""
        if status != 200 and isinstance(body, dict) and (body.get("message") or body.get("error")):
            why = f" ({_scrub(str(body.get('message') or body.get('error')))})"
        lines.append(f"{label[0]}: HTTP {status}; {_shape(body)}{why}")
        if status == 200:
            for name, items in _lists(body):
                if name not in ("positions", "fills"):
                    continue
                eth = [i for i in items if isinstance(i, dict) and _is_eth_perp(i)]
                fields = sorted(eth[0]) if eth else []
                lines.append(f"  ETH perpetual {name[:-1]} returned: {'yes' if eth else 'no'}"
                             + (f"; fields {', '.join(fields)}" if fields else ""))
                if eth and name == "positions":
                    parsed = position_from(eth[0])
                    lines.append(f"    values read: {_read(parsed, skip=('product', 'side', 'liquidation_from'))}")
                    for key, names in POSITION_FIELDS.items():
                        if parsed[key] is None:
                            field = next((n for n in names if n in eth[0]), names[0])
                            lines.append(f"    not read: {key} from {field} = {_describe(eth[0].get(field))}")
            if label[0] == "perpetuals portfolio summary" and isinstance(body, dict):
                lines.append(f"  values read: {_read(summary_from(body))}")
    held = account(client)["positions"]
    source = {"exchange": "the exchange's own", "buffer": "worked out from the portfolio's liquidation buffer", None: "none"}
    lines.append("dashboard liquidation price: " + ("; ".join(source[p.get("liquidation_from")] for p in held) if held else "no ETH perp position"))
    if held:
        lines.append("open trades rebuilt from fills: " + "; ".join(
            "all found" if p.get("trades_complete") else "older fills needed (shown as an earlier trade)" for p in held))
        lines += [f"  {line}" for line in _fills_against(client, held[0])]
    return lines


def _fills_against(client: Client, p: dict) -> list[str]:
    """How the fills read compare with the position, without amounts: kinds, units, a ratio and an age bucket."""
    meta: dict = {}
    fills = eth_fills(client, p["product"], meta)
    net = sum(f["size"] * (1 if f["side"] == "buy" else -1) for f in fills if f["size"])
    oldest = min((pd_time(f["time"]) for f in fills if f["time"]), default=None)
    age = None if oldest is None else (time.time() - oldest) / 86400
    bucket = "none" if age is None else "under 2 days" if age < 2 else "under 2 weeks" if age < 14 else "under 3 months" if age < 92 else "older"
    ratio = net / p["size"] if p.get("size") else None
    return [
        f"fills: product filter used {'yes' if meta.get('filtered') else 'no'}, every page read {'yes' if meta.get('ended') else 'no'}; "
        f"position product {p['product']}, fill products {', '.join(sorted({str(f['product']) for f in fills})) or 'none'}",
        f"fills: kinds {', '.join(sorted({f['type'] or '?' for f in fills})) or 'none'}; sides {', '.join(sorted({f['side'] or '?' for f in fills})) or 'none'}; "
        f"sized in quote currency {', '.join(sorted({'yes' if f['in_quote'] else 'no' for f in fills})) or 'n/a'}",
        f"fills: net bought / position size = {'n/a' if ratio is None else f'{ratio:.2g}'}; oldest fill read: {bucket}",
    ]


def pd_time(value: str) -> float:
    """Seconds since the epoch for an exchange time like 2026-10-25T17:02:11.123Z."""
    from datetime import datetime
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    return lines


def _describe(v, depth: int = 0) -> str:
    """A value's shape with nothing of its content: types, keys, and whether text is empty or a number."""
    if isinstance(v, dict):
        if depth > 2:
            return "{...}"
        return "{" + ", ".join(f"{k}: {_describe(x, depth + 1)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return f"a list of {len(v) and _describe(v[0], depth + 1) or 'nothing'}"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true/false"
    if isinstance(v, (int, float)):
        return "a number"
    if v == "":
        return "empty text"
    try:
        float(v)
        return "number text"
    except (TypeError, ValueError):
        return "other text"


def _read(parsed: dict, skip=()) -> str:
    """Which values came back as numbers: names and yes/no only."""
    return ", ".join(f"{k} {'yes' if v is not None else 'no'}" for k, v in parsed.items() if k not in skip)


def _is_eth_perp(item: dict) -> bool:
    product = str(item.get("product_id") or item.get("symbol") or "")
    return bool(ETH_PERP.match(product)) and "PERP" in product.upper()


def _lists(body) -> list[tuple[str, list]]:
    if isinstance(body, dict):
        return [(k, v) for k, v in body.items() if isinstance(v, list)]
    return []


def _portfolios(client: Client) -> list[dict]:
    status, body = client.get("/api/v3/brokerage/portfolios")
    if status != 200 or not isinstance(body, dict):
        return []
    return [p for p in body.get("portfolios", []) if isinstance(p, dict)]


def _endpoints(client: Client) -> list[tuple[str, tuple[str, dict | None]]]:
    out = [("/api/v3/brokerage/portfolios", ("portfolios", None))]
    for p in _portfolios(client):
        kind, uuid = str(p.get("type", "")), p.get("uuid")
        if uuid and "INTX" in kind.upper():
            out.append((f"/api/v3/brokerage/intx/positions/{uuid}", ("perpetuals positions (international)", None)))
            out.append((f"/api/v3/brokerage/intx/portfolio/{uuid}", ("perpetuals portfolio summary", None)))
    out.append(("/api/v3/brokerage/cfm/positions", ("futures positions (US)", None)))
    out.append(("/api/v3/brokerage/cfm/balance_summary", ("futures balance (US)", None)))
    out.append(("/api/v3/brokerage/orders/historical/fills", ("recent fills", {"limit": 50})))
    return out


# ---- the position itself, for the private dashboard -----------------------------------------------

def _num(v) -> float | None:
    """A number, or an amount object: {"value": ...}, or {"rawCurrency": {...}, "userNativeCurrency": {...}}, where
    the raw currency (the contract's USDC) is preferred to the account's display currency."""
    if isinstance(v, dict):
        for key in ("value", "amount"):
            if key in v:
                return _num(v[key])
        for key in ("rawCurrency", "raw_currency", "userNativeCurrency", "user_native_currency"):
            if isinstance(v.get(key), dict):
                return _num(v[key])
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(d: dict, *names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None


POSITION_FIELDS = {  # what the dashboard shows: the field names Coinbase's endpoints use for it, in order of preference
    "size": ("net_size", "number_of_contracts", "size"),
    "entry": ("entry_vwap", "vwap", "avg_entry_price", "entry_price"),
    "mark": ("mark_price", "current_price"),
    "liquidation": ("liquidation_price",),
    "unrealized": ("unrealized_pnl",),
    "leverage": ("leverage",),
    "notional": ("position_notional",),
    "maintenance": ("mm_notional",),
}


def position_from(item: dict) -> dict:
    side = str(_first(item, "position_side", "side") or "").upper()
    out = {"product": _first(item, "product_id", "symbol"), "side": "short" if "SHORT" in side else "long"}
    for key, names in POSITION_FIELDS.items():
        out[key] = _num(_first(item, *names))
    if out["size"] is not None and "SHORT" in side and out["size"] > 0:
        out["size"] = -out["size"]
    out["liquidation_from"] = "exchange" if out["liquidation"] is not None else None
    return out


def liquidation_from_buffer(p: dict, buffer: float | None) -> float | None:
    """A long's liquidation price from the portfolio's liquidation buffer (collateral above the maintenance
    margin), when the position is the portfolio's only one. Falling d in price costs size * d of equity and
    frees maintenance margin * size * d, so the buffer is gone after buffer / (size * (1 - m))."""
    size, mark = p.get("size"), p.get("mark")
    if buffer is None or not size or size <= 0 or not mark:
        return None
    m = p["maintenance"] / p["notional"] if p.get("maintenance") and p.get("notional") else 0.0
    return max(0.0, mark - buffer / (size * (1 - m)))


FILL_PAGES = 5  # pages of 100 fills: how far back the open trades are rebuilt from
EPS = 1e-6  # ETH: below the smallest order size (0.0001)


def summary_from(body: dict) -> dict:
    """The perpetuals portfolio: collateral and the liquidation buffer sit on the portfolio, totals on the summary."""
    portfolio = (body.get("portfolios") or [{}])[0]
    s = {**(body.get("summary") or {}), **(portfolio if isinstance(portfolio, dict) else {})}
    return {"collateral": _num(_first(s, "collateral", "total_balance")),
            "unrealized": _num(_first(s, "unrealized_pnl")),
            "liquidation_buffer": _num(_first(s, "liquidation_buffer")),
            "liquidation_percentage": _num(_first(s, "liquidation_percentage")),
            "total": _num(_first(s, "total_balance"))}


def fill_from(item: dict) -> dict:
    size, price = _num(item.get("size")), _num(item.get("price"))
    if size is not None and price and str(item.get("size_in_quote")).lower() == "true":
        size = size / price  # an order placed in USDC reports its size in USDC
    return {"time": _first(item, "trade_time", "created_time", "time"), "product": item.get("product_id"),
            "order": item.get("order_id"), "side": str(item.get("side", "")).lower(), "size": size, "price": price,
            "fee": _num(item.get("commission")), "type": str(item.get("trade_type") or ""),
            "in_quote": str(item.get("size_in_quote")).lower() == "true"}


def eth_fills(client: Client, product: str | None, meta: dict | None = None) -> list[dict]:
    """Up to FILL_PAGES pages of the ETH perpetual's fills, newest first (Coinbase's order). ``meta``, if given,
    is filled with whether the product filter was used and whether the pages reached the oldest fill."""
    def pages(params):
        out, cursor, ended = [], None, False
        for _ in range(FILL_PAGES):
            status, body = client.get("/api/v3/brokerage/orders/historical/fills", {**params, **({"cursor": cursor} if cursor else {})})
            if status != 200 or not isinstance(body, dict):
                break
            items = [i for i in body.get("fills", []) if isinstance(i, dict)]
            out += [fill_from(i) for i in items if _is_eth_perp(i)]
            cursor = body.get("cursor")
            if not cursor or not items:
                ended = True
                break
        return out, ended
    fills, ended = pages({"product_ids": [product], "limit": 100}) if product else ([], False)
    filtered = bool(fills)
    if not fills:  # if the filter is not honoured for perps, read everything and keep ETH
        fills, ended = pages({"limit": 100})
    if meta is not None:
        meta.update(filtered=filtered, ended=ended)
    return fills


def fills_for(fills: list[dict], position: dict, eth_positions: int) -> list[dict]:
    """The fills of one position. Coinbase can name a perpetual differently in its fills and its positions, so
    with a single ETH perpetual position every ETH perpetual fill is its own; with several, names must match."""
    return fills if eth_positions == 1 else [f for f in fills if f["product"] == position["product"]]


def open_trades(fills: list[dict], net_size: float | None) -> dict:
    """The orders that make up today's position, oldest first, each with what is still open of it.

    Fills are grouped by order (one order can fill in many pieces) and matched first in, first out: a sell
    closes the oldest buys first. Whatever the fetched fills do not explain (the position held before the
    oldest of them) is an 'earlier' trade with no price, and ``complete`` is then false."""
    orders: dict[str, dict] = {}
    for f in sorted((f for f in fills if f["size"] and f["price"] and f["time"]), key=lambda f: f["time"]):
        q = f["size"] * (1 if f["side"] == "buy" else -1)
        o = orders.setdefault(f["order"] or f["time"], {"time": f["time"], "qty": 0.0, "gross": 0.0, "cost": 0.0, "fee": 0.0})
        o["qty"] += q
        o["gross"] += abs(q)
        o["cost"] += abs(q) * f["price"]
        o["fee"] += f["fee"] or 0.0
    net = net_size or 0.0
    earlier = net - sum(o["qty"] for o in orders.values())
    lots = [{"time": None, "price": None, "size": earlier, "opened": earlier, "fee": None}] if abs(earlier) > EPS else []
    for o in orders.values():
        q = o["qty"]
        if abs(q) <= EPS:
            continue
        while abs(q) > EPS and lots and (lots[0]["size"] > 0) != (q > 0):  # q works against the open lots: close oldest first
            lot = lots[0]
            if abs(lot["size"]) <= abs(q) + EPS:
                q += lot["size"]
                lots.pop(0)
            else:
                lot["size"] += q
                q = 0.0
        if abs(q) > EPS:
            lots.append({"time": o["time"], "price": o["cost"] / o["gross"], "size": q, "opened": o["qty"], "fee": o["fee"]})
    return {"trades": lots, "complete": abs(earlier) <= EPS}


def account(client: Client) -> dict:
    """Your ETH perpetual positions, each with the trades still open in it, and the perpetuals portfolio's collateral."""
    positions, summary, held = [], {}, 0
    for p in _portfolios(client):
        if "INTX" not in str(p.get("type", "")).upper() or not p.get("uuid"):
            continue
        status, body = client.get(f"/api/v3/brokerage/intx/positions/{p['uuid']}")
        if status == 200 and isinstance(body, dict):
            items = [i for i in body.get("positions", []) if isinstance(i, dict)]
            held += len(items)
            positions += [position_from(i) for i in items]
        status, body = client.get(f"/api/v3/brokerage/intx/portfolio/{p['uuid']}")
        if status == 200 and isinstance(body, dict):
            summary = summary_from(body)
    status, body = client.get("/api/v3/brokerage/cfm/positions")
    if status == 200 and isinstance(body, dict):
        positions += [position_from(i) for i in body.get("positions", []) if isinstance(i, dict)]
    eth = [p for p in positions if _is_eth_perp({"product_id": p.get("product")})]
    for p in eth:
        rebuilt = open_trades(fills_for(eth_fills(client, p["product"]), p, len(eth)), p["size"])
        p["trades"], p["trades_complete"] = rebuilt["trades"], rebuilt["complete"]
    if held == 1 and len(eth) == 1 and eth[0]["liquidation"] is None and summary:
        eth[0]["liquidation"] = liquidation_from_buffer(eth[0], summary.get("liquidation_buffer"))
        eth[0]["liquidation_from"] = "buffer" if eth[0]["liquidation"] is not None else None
    return {"source": "coinbase", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "positions": eth, "summary": summary or None}
