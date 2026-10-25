import base64
import json

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from ethtracker import coinbase
from ethtracker.cli import main
from ethtracker.data import DataError

from tests.fakes import FakeDashboard
from tests.test_cli import cloud_env, fake_deribit

UUID = "0f1e2d3c-4b5a-6978-8a9b-0c1d2e3f4a5b"


def ec_key_json() -> tuple[str, ec.EllipticCurvePrivateKey]:
    private = ec.generate_private_key(ec.SECP256R1())
    pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                serialization.NoEncryption()).decode()
    return json.dumps({"name": "organizations/o/apiKeys/k", "privateKey": pem}), private


def ed_key_json() -> tuple[str, ed25519.Ed25519PrivateKey]:
    private = ed25519.Ed25519PrivateKey.generate()
    seed = private.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    public = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return json.dumps({"id": "key-id", "privateKey": base64.b64encode(seed + public).decode()}), private


def parts(jwt: str):
    pad = lambda s: s + "=" * (-len(s) % 4)  # noqa: E731
    h, c, s = jwt.split(".")
    return json.loads(base64.urlsafe_b64decode(pad(h))), json.loads(base64.urlsafe_b64decode(pad(c))), \
        base64.urlsafe_b64decode(pad(s)), f"{h}.{c}".encode()


def test_an_ec_key_signs_es256_tokens_its_public_key_verifies():
    text, private = ec_key_json()
    key = coinbase.load_key(text)
    assert key.alg == "ES256" and key.id == "organizations/o/apiKeys/k"
    header, claims, sig, signed = parts(coinbase.token(key, "GET", "/api/v3/brokerage/portfolios?x=1", now=1000))
    assert header["alg"] == "ES256" and header["kid"] == key.id and len(header["nonce"]) == 32
    assert claims == {"sub": key.id, "iss": "cdp", "nbf": 1000, "exp": 1120, "uri": "GET api.coinbase.com/api/v3/brokerage/portfolios"}
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
    private.public_key().verify(der, signed, ec.ECDSA(hashes.SHA256()))  # raises if wrong


def test_an_ed25519_key_signs_eddsa_tokens_and_the_other_claim_style_uses_uris():
    text, private = ed_key_json()
    key = coinbase.load_key(text)
    assert key.alg == "EdDSA" and key.id == "key-id"
    header, claims, sig, signed = parts(coinbase.token(key, "GET", "/api/v3/brokerage/key_permissions", style="uris"))
    assert header["alg"] == "EdDSA" and claims["uris"] == ["GET api.coinbase.com/api/v3/brokerage/key_permissions"]
    private.public_key().verify(sig, signed)


def test_a_key_copied_with_real_line_breaks_or_as_two_pasted_parts_still_loads():
    text, _ = ec_key_json()
    data = json.loads(text)
    raw_newlines = '{\n  "name": "%s",\n  "privateKey": "%s"\n}\n' % (data["name"], data["privateKey"])
    assert "\n" in data["privateKey"] and coinbase.load_key(raw_newlines).alg == "ES256"
    two_parts = f"{data['name']}\n{data['privateKey']}"
    assert coinbase.load_key(two_parts).id == data["name"]


def test_a_broken_key_names_what_is_missing_but_never_the_secret():
    with pytest.raises(DataError, match="not a key file: 8 characters") as exc:
        coinbase.load_key("s3cretxx")
    assert "s3cret" not in str(exc.value) and "private key block: no" in str(exc.value)
    with pytest.raises(DataError, match="fields found: privateKey") as exc:
        coinbase.load_key(json.dumps({"privateKey": "s3cret"}))
    assert "s3cret" not in str(exc.value)


class FakeClient:
    """Answers like Coinbase's App API, with values a public log must never show."""
    style = "uri"

    def __init__(self, positions=True):
        self.positions = positions

    def get(self, path, params=None):
        if path.endswith("key_permissions"):
            return 200, {"can_view": True, "can_trade": False, "can_transfer": False, "portfolio_uuid": UUID, "portfolio_type": "INTX"}
        if path.endswith("/portfolios"):
            return 200, {"portfolios": [{"name": "Perps", "uuid": UUID, "type": "INTX"}, {"name": "Main", "uuid": "x", "type": "DEFAULT"}]}
        if "/intx/positions/" in path:
            items = [{"product_id": "ETH-PERP-INTX", "position_side": "POSITION_SIDE_LONG", "net_size": "4.5321",
                      "entry_vwap": {"value": "2311.17", "currency": "USDC"}, "mark_price": {"value": "2340.55"},
                      "liquidation_price": {"value": "1262.40"}, "unrealized_pnl": {"value": "133.15"}, "leverage": "2.1",
                      "position_notional": {"value": "10607.43"}}] if self.positions else []
            return 200, {"positions": items, "summary": {}}
        if "/intx/portfolio/" in path:
            return 200, {"summary": {"collateral": {"value": "5004.99"}, "unrealized_pnl": {"value": "133.15"},
                                     "liquidation_buffer": {"value": "4521.00"}}}
        if "/cfm/" in path:
            return 404, {"error": "NOT_FOUND", "message": f"no futures portfolio {UUID}"}
        if path.endswith("/fills"):
            return 200, {"fills": [{"product_id": "ETH-PERP-INTX", "side": "BUY", "size": "4.5321", "price": "2311.17",
                                    "commission": "5.23", "trade_time": "2026-10-25T17:02:11Z"},
                                   {"product_id": "SOL-USD", "side": "BUY", "size": "9", "price": "180", "commission": "1"}]}
        return 404, None


def test_the_probe_reports_shapes_and_yes_or_no_never_amounts_ids_or_other_products():
    lines = "\n".join(coinbase.probe(FakeClient()))
    assert "view True, trade False, transfer False" in lines
    assert "ETH perpetual position returned: yes" in lines and "ETH perpetual fill returned: yes" in lines
    assert "liquidation_price" in lines  # field names are fine
    for secret in ("4.5321", "2311", "1262", "5004", UUID, "SOL", "<id>x"):
        assert secret not in lines, secret
    assert "no futures portfolio" in lines and UUID not in lines  # error messages are scrubbed of identifiers


def test_a_key_that_can_trade_is_called_out():
    class Trading(FakeClient):
        def get(self, path, params=None):
            if path.endswith("key_permissions"):
                return 200, {"can_view": True, "can_trade": True, "can_transfer": False}
            return super().get(path, params)
    assert any("WARNING" in line for line in coinbase.probe(Trading()))


def test_the_account_holds_the_eth_position_its_open_trades_and_the_collateral():
    acct = coinbase.account(FakeClient())
    (p,) = acct["positions"]
    assert p["side"] == "long" and p["size"] == pytest.approx(4.5321) and p["entry"] == pytest.approx(2311.17)
    assert p["liquidation"] == pytest.approx(1262.40) and p["mark"] == pytest.approx(2340.55) and p["leverage"] == 2.1
    assert acct["summary"]["collateral"] == pytest.approx(5004.99)
    assert "fills" not in acct
    (t,) = p["trades"]
    assert p["trades_complete"] and t["time"] == "2026-10-25T17:02:11Z" and t["size"] == pytest.approx(4.5321)
    assert t["price"] == pytest.approx(2311.17)
    assert coinbase.account(FakeClient(positions=False))["positions"] == []


def test_a_401_is_retried_once_with_the_other_claim_style(monkeypatch):
    text, _ = ec_key_json()
    client = coinbase.Client(coinbase.load_key(text))
    seen = []

    class Response:
        def __init__(self, status):
            self.status_code = status

        def json(self):
            return {}

    def fake_get(url, params=None, timeout=None, headers=None):
        _, claims, _, _ = parts(headers["Authorization"].split(" ", 1)[1])
        seen.append("uri" if "uri" in claims else "uris")
        return Response(401 if len(seen) == 1 else 200)

    monkeypatch.setattr(coinbase.requests, "get", fake_get)
    assert client.get("/api/v3/brokerage/portfolios")[0] == 200
    assert seen == ["uri", "uris"] and client.style == "uris"


def test_account_check_needs_the_key(capsys, monkeypatch):
    monkeypatch.delenv(coinbase.ENV, raising=False)
    assert main(["account-check"]) == 2
    assert coinbase.ENV in capsys.readouterr().err


def test_the_daily_job_sends_the_account_to_the_dashboard_and_nothing_of_it_to_the_logs(capsys, monkeypatch, candles_csv):
    text, _ = ec_key_json()
    monkeypatch.setenv(coinbase.ENV, text)
    monkeypatch.setattr(coinbase, "Client", lambda key: FakeClient())
    cloud_env(monkeypatch)
    dash = FakeDashboard().install(monkeypatch)
    fake_deribit(monkeypatch)
    assert main(["--provider", "csv", "--csv", str(candles_csv), "cloud-run"]) == 0
    captured = capsys.readouterr()
    (snap,) = dash.snapshots
    assert snap["account"]["positions"][0]["liquidation"] == pytest.approx(1262.40)
    (trade,) = snap["account"]["positions"][0]["trades"]
    assert "fills" not in snap["account"] and trade["signals"]["score"] is not None and trade["signals"]["components"]
    for secret in ("4.5321", "1262", "2311", "5004", UUID):
        assert secret not in captured.out + captured.err, secret


def test_a_bare_ed25519_secret_needs_its_id_and_works_with_it(monkeypatch):
    text, private = ed_key_json()
    secret = json.loads(text)["privateKey"]
    assert len(secret) == 88
    with pytest.raises(DataError, match="secret on its own") as exc:
        coinbase.load_key(secret)
    assert secret not in str(exc.value) and coinbase.ENV_ID in str(exc.value)
    monkeypatch.setenv(coinbase.ENV, secret)
    monkeypatch.setenv(coinbase.ENV_ID, " 7d9f0c2e-key-id \n")
    key = coinbase.env_key()
    assert key.alg == "EdDSA" and key.id == "7d9f0c2e-key-id"
    _, _, sig, signed = parts(coinbase.token(key, "GET", "/api/v3/brokerage/portfolios"))
    private.public_key().verify(sig, signed)


def amount(value):
    return {"userNativeCurrency": {"value": str(float(value) * 0.79), "currency": "GBP"},
            "rawCurrency": {"value": value, "currency": "USDC"}}


class NestedClient(FakeClient):
    """Amounts as objects in two currencies; collateral on the portfolio, totals on the summary."""

    def get(self, path, params=None):
        if "/intx/positions/" in path:
            return 200, {"positions": [{"product_id": "ETH-PERP-INTX", "position_side": "POSITION_SIDE_LONG", "net_size": "4.5321",
                                        "entry_vwap": amount("2311.17"), "vwap": amount("2320.00"), "mark_price": amount("2340.55"),
                                        "liquidation_price": amount("1262.40"), "unrealized_pnl": amount("133.15"), "leverage": "2.1",
                                        "position_notional": amount("10607.43"), "mm_notional": amount("212.15")}]}
        if "/intx/portfolio/" in path:
            return 200, {"portfolios": [{"collateral": "5004.99", "liquidation_buffer": "4521.00", "liquidation_percentage": "9.6",
                                         "unrealized_pnl": amount("133.15")}],
                         "summary": {"unrealized_pnl": amount("133.15"), "total_balance": amount("5138.14")}}
        return super().get(path, params)


def test_amounts_nested_by_currency_read_in_the_contracts_currency_and_the_probe_says_which_values_came_back():
    acct = coinbase.account(NestedClient())
    (p,) = acct["positions"]
    assert p["entry"] == pytest.approx(2311.17) and p["liquidation"] == pytest.approx(1262.40)
    assert p["mark"] == pytest.approx(2340.55) and p["maintenance"] == pytest.approx(212.15)
    assert acct["summary"] == {"collateral": pytest.approx(5004.99), "unrealized": pytest.approx(133.15),
                               "liquidation_buffer": pytest.approx(4521.0), "liquidation_percentage": pytest.approx(9.6),
                               "total": pytest.approx(5138.14)}
    lines = "\n".join(coinbase.probe(NestedClient()))
    assert "values read: size yes, entry yes, mark yes, liquidation yes" in lines
    assert "values read: collateral yes, unrealized yes, liquidation_buffer yes" in lines
    for secret in ("4.5321", "2311", "1262", "5004", "4521", "9.6", UUID):
        assert secret not in lines, secret


class CrossMarginClient(NestedClient):
    """A cross-margined position whose own liquidation price and leverage come back empty."""

    def get(self, path, params=None):
        status, body = super().get(path, params)
        if "/intx/positions/" in path:
            body["positions"][0]["liquidation_price"] = {"rawCurrency": {"value": "", "currency": "USDC"}}
            body["positions"][0]["leverage"] = ""
        return status, body


def test_a_missing_liquidation_price_comes_from_the_buffer_and_the_probe_describes_the_field_without_values():
    (p,) = coinbase.account(CrossMarginClient())["positions"]
    m = 212.15 / 10607.43
    assert p["liquidation_from"] == "buffer"
    assert p["liquidation"] == pytest.approx(2340.55 - 4521.0 / (4.5321 * (1 - m)))
    # a price that far down eats exactly the buffer: equity lost less maintenance freed
    d = 2340.55 - p["liquidation"]
    assert 4.5321 * d - m * 4.5321 * d == pytest.approx(4521.0)
    lines = "\n".join(coinbase.probe(CrossMarginClient()))
    assert "not read: liquidation from liquidation_price = {rawCurrency: {value: empty text, currency: other text}}" in lines
    assert "not read: leverage from leverage = empty text" in lines
    assert "dashboard liquidation price: worked out from the portfolio's liquidation buffer" in lines
    assert "dashboard liquidation price: the exchange's own" in "\n".join(coinbase.probe(FakeClient()))
    for secret in ("4.5321", "2311", "4521", "USDC", UUID):
        assert secret not in lines, secret


def test_no_buffer_estimate_when_the_portfolio_holds_other_positions_too():
    class TwoPositions(CrossMarginClient):
        def get(self, path, params=None):
            status, body = super().get(path, params)
            if "/intx/positions/" in path:
                body["positions"].append({"product_id": "BTC-PERP-INTX", "position_side": "POSITION_SIDE_LONG", "net_size": "0.1"})
            return status, body
    (p,) = coinbase.account(TwoPositions())["positions"]
    assert p["liquidation"] is None and p["liquidation_from"] is None


def fill(time, side, size, price, order=None, quote=False):
    return {"time": time, "product": "ETH-PERP-INTX", "order": order or time, "side": side, "size": size, "price": price, "fee": 0.1}


def test_open_trades_group_fills_by_order_and_close_the_oldest_first():
    fills = [
        fill("2026-09-01T10:00:00Z", "buy", 1.0, 2000.0, "a"), fill("2026-09-01T10:00:01Z", "buy", 1.0, 2100.0, "a"),
        fill("2026-09-08T10:00:00Z", "buy", 1.0, 1900.0, "b"),
        fill("2026-09-15T10:00:00Z", "sell", 1.5, 2200.0, "c"),
        fill("2026-09-22T10:00:00Z", "buy", 0.5, 2300.0, "d"),
    ]
    out = coinbase.open_trades(fills, 2.0)
    assert out["complete"]
    assert [(t["time"][:10], round(t["size"], 6), round(t["price"], 2)) for t in out["trades"]] == [
        ("2026-09-01", 0.5, 2050.0), ("2026-09-08", 1.0, 1900.0), ("2026-09-22", 0.5, 2300.0)]
    assert out["trades"][0]["opened"] == pytest.approx(2.0)


def test_a_round_trip_to_flat_leaves_only_the_trades_since():
    fills = [fill("2026-08-01T00:00:00Z", "buy", 2.0, 3000.0), fill("2026-08-20T00:00:00Z", "sell", 2.0, 2500.0),
             fill("2026-10-25T17:00:00Z", "buy", 1.0, 2311.0)]
    out = coinbase.open_trades(fills, 1.0)
    assert out["complete"] and [t["time"][:10] for t in out["trades"]] == ["2026-10-25"]


def test_a_position_older_than_the_fills_fetched_shows_as_an_earlier_trade():
    out = coinbase.open_trades([fill("2026-10-25T17:00:00Z", "buy", 1.0, 2311.0)], 3.0)
    assert not out["complete"]
    assert out["trades"][0]["time"] is None and out["trades"][0]["size"] == pytest.approx(2.0)
    assert sum(t["size"] for t in out["trades"]) == pytest.approx(3.0)


def test_a_fill_sized_in_usdc_is_converted_to_eth():
    f = coinbase.fill_from({"product_id": "ETH-PERP-INTX", "side": "BUY", "size": "2311.17", "price": "2311.17", "size_in_quote": True,
                            "trade_time": "2026-10-25T17:02:11Z", "order_id": "o"})
    assert f["size"] == pytest.approx(1.0) and f["order"] == "o"


def test_fills_are_paged_until_the_cursor_runs_out():
    class Paged(FakeClient):
        seen = []

        def get(self, path, params=None):
            if path.endswith("/fills"):
                self.seen.append(dict(params))
                page = int(params.get("cursor") or 0)
                return 200, {"fills": [{"product_id": "ETH-PERP-INTX", "side": "BUY", "size": "1", "price": "2000",
                                        "trade_time": f"2026-10-2{page}T00:00:00Z", "order_id": str(page)}],
                             "cursor": str(page + 1) if page < 2 else ""}
            return super().get(path, params)
    client = Paged()
    fills = coinbase.eth_fills(client, "ETH-PERP-INTX")
    assert len(fills) == 3 and client.seen[0]["product_ids"] == ["ETH-PERP-INTX"] and client.seen[-1]["cursor"] == "2"
