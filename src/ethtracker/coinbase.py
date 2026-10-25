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
HOST = "api.coinbase.com"
BASE = f"https://{HOST}"
ETH_PERP = re.compile(r"^ETH[-_]")  # ETH-PERP-INTX, ETH-PERP, ETH_USDC-PERPETUAL ...
UUIDISH = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}|[0-9a-fA-F]{24,}")


@dataclass(frozen=True)
class Key:
    id: str
    private: object
    alg: str  # ES256 or EdDSA


def load_key(text: str) -> Key:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise DataError(f"{ENV} is not JSON: paste the whole downloaded key file") from exc
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
    return load_key(text) if text else None


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
    return lines


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
    if isinstance(v, dict):
        v = v.get("value", v.get("amount"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first(d: dict, *names):
    for n in names:
        if n in d and d[n] not in (None, ""):
            return d[n]
    return None


def position_from(item: dict) -> dict:
    side = str(_first(item, "position_side", "side") or "").upper()
    size = _num(_first(item, "net_size", "number_of_contracts", "size"))
    if size is not None and "SHORT" in side and size > 0:
        size = -size
    return {
        "product": _first(item, "product_id", "symbol"),
        "side": "short" if "SHORT" in side else "long",
        "size": size,
        "entry": _num(_first(item, "entry_vwap", "vwap", "avg_entry_price", "entry_price")),
        "mark": _num(_first(item, "mark_price", "current_price")),
        "liquidation": _num(_first(item, "liquidation_price")),
        "unrealized": _num(_first(item, "unrealized_pnl")),
        "leverage": _num(_first(item, "leverage")),
        "notional": _num(_first(item, "position_notional")),
        "maintenance": _num(_first(item, "mm_notional")),
    }


def fill_from(item: dict) -> dict:
    return {"time": _first(item, "trade_time", "created_time", "time"), "product": item.get("product_id"),
            "side": str(item.get("side", "")).lower(), "size": _num(item.get("size")), "price": _num(item.get("price")),
            "fee": _num(item.get("commission"))}


def account(client: Client) -> dict:
    """Your ETH perpetual positions, the perpetuals portfolio's collateral and recent ETH perpetual fills."""
    positions, summary = [], {}
    for p in _portfolios(client):
        if "INTX" not in str(p.get("type", "")).upper() or not p.get("uuid"):
            continue
        status, body = client.get(f"/api/v3/brokerage/intx/positions/{p['uuid']}")
        if status == 200 and isinstance(body, dict):
            positions += [position_from(i) for i in body.get("positions", []) if isinstance(i, dict)]
        status, body = client.get(f"/api/v3/brokerage/intx/portfolio/{p['uuid']}")
        if status == 200 and isinstance(body, dict):
            s = body.get("summary") or (body.get("portfolios") or [{}])[0] or {}
            summary = {"collateral": _num(_first(s, "collateral", "total_balance")),
                       "unrealized": _num(_first(s, "unrealized_pnl")),
                       "liquidation_buffer": _num(_first(s, "liquidation_buffer")),
                       "liquidation_percentage": _num(_first(s, "liquidation_percentage"))}
    status, body = client.get("/api/v3/brokerage/cfm/positions")
    if status == 200 and isinstance(body, dict):
        positions += [position_from(i) for i in body.get("positions", []) if isinstance(i, dict)]
    status, body = client.get("/api/v3/brokerage/orders/historical/fills", {"limit": 50})
    fills = []
    if status == 200 and isinstance(body, dict):
        fills = [fill_from(i) for i in body.get("fills", []) if isinstance(i, dict) and _is_eth_perp(i)]
    eth = [p for p in positions if _is_eth_perp({"product_id": p.get("product")})]
    return {"source": "coinbase", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "positions": eth, "summary": summary or None, "fills": fills[:20]}
