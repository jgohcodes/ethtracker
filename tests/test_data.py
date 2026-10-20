from datetime import datetime, timedelta

import pandas as pd
import pytest

from ethtracker import data


def test_parse_coinbase_candles_maps_columns_sorts_and_dedupes():
    rows = [
        [1700086400, 95, 120, 105, 110, 7.0],  # newest first, as the API returns
        [1700000000, 90, 110, 100, 105, 5.0],
        [1700000000, 90, 110, 100, 105, 5.0],  # duplicate boundary candle
    ]
    df = data.parse_coinbase_candles(rows)
    assert list(df.columns) == data.COLUMNS
    assert len(df) == 2 and df.index.is_monotonic_increasing
    first = df.iloc[0]
    assert (first.open, first.high, first.low, first.close) == (100, 110, 90, 105)


def test_parse_kraken_ohlc_converts_strings_and_ignores_pair_key():
    payload = {
        "error": [],
        "result": {
            "XETHZUSD": [[1700000000, "100", "110", "90", "105", "102", "5.0", 10]],
            "last": 1700000000,
        },
    }
    df = data.parse_kraken_ohlc(payload)
    row = df.iloc[0]
    assert (row.open, row.high, row.low, row.close, row.volume) == (100, 110, 90, 105, 5.0)


def test_kraken_error_payload_raises():
    with pytest.raises(data.DataError, match="EQuery"):
        data.parse_kraken_ohlc({"error": ["EQuery:Unknown asset pair"], "result": {}})


def test_parse_coinbase_spot_accepts_documented_shape_with_or_without_base():
    assert data.parse_coinbase_spot({"data": {"amount": "3000.12", "currency": "USD"}}) == 3000.12
    assert data.parse_coinbase_spot({"data": {"base": "ETH", "currency": "USD", "amount": "2999"}}) == 2999.0


@pytest.mark.parametrize("payload", [{}, {"data": {}}, {"data": {"amount": "n/a"}}, {"errors": [{"id": "x"}]}, None])
def test_parse_coinbase_spot_rejects_unexpected_payloads(payload):
    with pytest.raises(data.DataError, match="unexpected Coinbase price payload"):
        data.parse_coinbase_spot(payload)


def test_coinbase_spot_uses_the_prices_api_not_the_exchange_ticker(monkeypatch):
    seen = []
    monkeypatch.setattr(data, "_get_json", lambda url, params=None: seen.append(url) or {"data": {"amount": "1.5"}})
    assert data.Coinbase().spot() == 1.5
    assert seen == ["https://api.coinbase.com/v2/prices/ETH-USD/spot"]


def test_empty_candles_raise():
    with pytest.raises(data.DataError):
        data.parse_coinbase_candles([])


def test_csv_provider_reads_iso_dates_and_epoch_seconds(tmp_path):
    iso = tmp_path / "iso.csv"
    iso.write_text("Date,Open,High,Low,Close\n2026-01-01,1,3,0.5,2\n2026-01-02,2,4,1.5,3\n")
    provider = data.get_provider("csv", iso)
    assert provider.spot() == 3.0
    assert len(provider.candles(1)) == 1
    assert provider.candles(10)["volume"].eq(0).all()

    epoch = tmp_path / "epoch.csv"
    epoch.write_text("time,open,high,low,close,volume\n1700000000,1,3,0.5,2,9\n")
    assert data.get_provider("csv", epoch).candles(5).index[0] == pd.Timestamp("2023-11-14 22:13:20", tz="UTC")


def test_csv_provider_reports_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,close\n2026-01-01,2\n")
    with pytest.raises(data.DataError, match="missing"):
        data.get_provider("csv", bad).spot()


def test_get_provider_validation():
    with pytest.raises(data.DataError):
        data.get_provider("csv")
    with pytest.raises(data.DataError):
        data.get_provider("nope")


def test_coinbase_paginates_past_the_300_candle_limit(monkeypatch):
    calls = []

    def fake_get_json(url, params=None):
        calls.append(url)
        if "/v2/prices/" in url:
            return {"data": {"amount": "3000.5", "currency": "USD"}}
        start, end = (datetime.fromisoformat(params[k]) for k in ("start", "end"))
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        rows = []
        while day < end:
            rows.append([int(day.timestamp()), 90.0, 110.0, 100.0, 105.0, 1.0])
            day += timedelta(days=1)
        return rows[::-1]

    monkeypatch.setattr(data, "_get_json", fake_get_json)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    provider = data.Coinbase()
    assert provider.spot() == 3000.5
    df = provider.candles(350)
    candle_calls = [c for c in calls if c.endswith("/candles")]
    assert len(candle_calls) == 2
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert 349 <= len(df) <= 352
