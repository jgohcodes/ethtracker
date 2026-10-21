from datetime import datetime, timezone

import pandas as pd
import pytest

from ethtracker import data, derivs

# the shape documented for Deribit's public/ticker: a JSON-RPC envelope with the figures under "result"
TICKER = {
    "jsonrpc": "2.0",
    "result": {
        "instrument_name": "ETH_USDC-PERPETUAL",
        "open_interest": 123456789.0,
        "funding_8h": 0.0001,
        "current_funding": 0.00012,
        "mark_price": 3001.5,
        "index_price": 3000.9,
        "timestamp": 1760961600000,
    },
}


def test_parse_deribit_ticker_reads_the_documented_fields():
    row = derivs.parse_deribit_ticker(TICKER)
    assert row["open_interest"] == 123456789.0 and row["funding_8h"] == 0.0001
    assert row["current_funding"] == 0.00012 and row["mark_price"] == 3001.5 and row["index_price"] == 3000.9
    assert row["time"] == pd.Timestamp(1760961600000, unit="ms", tz="UTC").isoformat()


def test_parse_deribit_ticker_falls_back_to_the_clock_and_tolerates_missing_optionals():
    payload = {"result": {"open_interest": 5, "funding_8h": 0.0}}
    now = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
    row = derivs.parse_deribit_ticker(payload, now=now)
    assert row["time"] == "2026-10-20T12:00:00+00:00"
    assert pd.isna(row["mark_price"]) and pd.isna(row["current_funding"])


@pytest.mark.parametrize("payload", [
    {}, {"error": {"code": 13009, "message": "invalid"}}, {"result": {}},
    {"result": {"open_interest": 1}}, {"result": {"open_interest": "n/a", "funding_8h": 0.0}}, None,
])
def test_unexpected_ticker_payloads_fail_loudly_instead_of_producing_numbers(payload):
    with pytest.raises(data.DataError, match="unexpected Deribit ticker payload"):
        derivs.parse_deribit_ticker(payload)


def test_fetch_asks_the_public_ticker_for_the_instrument(monkeypatch):
    seen = {}

    def fake(url, params=None):
        seen.update(url=url, params=params)
        return TICKER

    monkeypatch.setattr(derivs, "_get_json", fake)
    assert derivs.fetch_deribit_snapshot("ETH_USDC-PERPETUAL")["funding_8h"] == 0.0001
    assert seen == {"url": "https://www.deribit.com/api/v2/public/ticker", "params": {"instrument_name": "ETH_USDC-PERPETUAL"}}


def test_snapshots_append_skip_duplicates_and_load_back(tmp_path):
    path = derivs.recorded_path(tmp_path / "derivs", "ETH_USDC-PERPETUAL")
    first = derivs.parse_deribit_ticker(TICKER)
    assert derivs.append_snapshot(path, first) is True
    assert derivs.append_snapshot(path, first) is False  # same timestamp
    later = dict(first, time=pd.Timestamp(1760965200000, unit="ms", tz="UTC").isoformat(), open_interest=130000000.0)
    assert derivs.append_snapshot(path, later) is True
    frame = derivs.load_csv(path)
    assert len(frame) == 2 and frame.index.is_monotonic_increasing and str(frame.index.tz) == "UTC"
    assert list(frame["open_interest"]) == [123456789.0, 130000000.0]
    assert path.read_text().splitlines()[0] == "time,open_interest,funding_8h,current_funding,mark_price,index_price"


def test_load_csv_accepts_any_known_columns_and_ignores_the_rest(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("date,funding_8h,long_liq_usd,whatever\n2026-10-01,0.0001,5,x\n2026-10-02,0.0002,6,y\n")
    frame = derivs.load_csv(path)
    assert list(frame.columns) == ["funding_8h", "long_liq_usd"] and len(frame) == 2


def test_load_csv_reads_epoch_seconds(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("time,open_interest\n1760961600,10\n1761048000,12\n")
    assert derivs.load_csv(path).index[0] == pd.Timestamp("2025-10-20 12:00:00", tz="UTC")


def test_load_csv_rejects_files_with_no_usable_columns(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("date,price\n2026-10-01,3000\n")
    with pytest.raises(data.DataError, match="at least one of"):
        derivs.load_csv(path)
    with pytest.raises(data.DataError, match="cannot read"):
        derivs.load_csv(tmp_path / "missing.csv")
