"""Price sources. Each returns a spot price and daily candles (UTC index; open/high/low/close/volume).

Coinbase and Kraken use their public, unauthenticated REST endpoints. The CSV
provider reads a local file, so everything works offline. Prices are spot, not a
perpetual's mark price, which is what liquidation actually uses.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def spot(self) -> float: ...

    def candles(self, days: int) -> pd.DataFrame: ...


def _get_json(url: str, params: dict | None = None):
    try:
        response = requests.get(url, params=params, timeout=15, headers={"User-Agent": "ethtracker/0.1"})
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataError(f"GET {url} failed: {exc}") from exc


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[COLUMNS].astype(float)
    if df.empty:
        raise DataError("provider returned no candles")
    return df


def parse_coinbase_candles(rows: list) -> pd.DataFrame:
    """Coinbase Exchange rows are [time, low, high, open, close, volume], newest first."""
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return _frame(df)


def parse_kraken_ohlc(payload: dict) -> pd.DataFrame:
    """Kraken rows are [time, open, high, low, close, vwap, volume, count]; the pair key varies."""
    if payload.get("error"):
        raise DataError(f"Kraken error: {payload['error']}")
    result = payload["result"]
    key = next(k for k in result if k != "last")
    df = pd.DataFrame(result[key], columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return _frame(df)


def parse_coinbase_spot(payload: dict) -> float:
    """Prices API spot payload: {"data": {"amount": "...", "currency": "USD"}} (``base`` is not relied on)."""
    try:
        return float(payload["data"]["amount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"unexpected Coinbase price payload: {payload!r}") from exc


class Coinbase:
    """Spot from the Coinbase Prices API (public, no key); candles from Coinbase Exchange's public endpoint.

    The Prices API returns a single price with no highs or lows, so it cannot feed ATR or
    adverse-move statistics; those need candles.
    """

    name = "coinbase"
    base = "https://api.exchange.coinbase.com"
    max_candles = 300

    def __init__(self, product: str = "ETH-USD"):
        self.product = product
        self.spot_url = f"https://api.coinbase.com/v2/prices/{product}/spot"

    def spot(self) -> float:
        return parse_coinbase_spot(_get_json(self.spot_url))

    def candles(self, days: int) -> pd.DataFrame:
        rows: list = []
        end = datetime.now(timezone.utc)
        remaining = days
        while remaining > 0:
            span = min(remaining, self.max_candles)
            start = end - timedelta(days=span)
            rows += _get_json(
                f"{self.base}/products/{self.product}/candles",
                {"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()},
            )
            end, remaining = start, remaining - span
            if remaining > 0:
                time.sleep(0.25)
        return parse_coinbase_candles(rows)


class Kraken:
    name = "kraken"
    base = "https://api.kraken.com/0/public"
    pair = "ETHUSD"

    def spot(self) -> float:
        payload = _get_json(f"{self.base}/Ticker", {"pair": self.pair})
        if payload.get("error"):
            raise DataError(f"Kraken error: {payload['error']}")
        return float(next(iter(payload["result"].values()))["c"][0])

    def candles(self, days: int) -> pd.DataFrame:
        since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        payload = _get_json(f"{self.base}/OHLC", {"pair": self.pair, "interval": 1440, "since": since})
        return parse_kraken_ohlc(payload).tail(days)


class Csv:
    """Daily candles from a file with time/date, open, high, low, close[, volume] columns."""

    name = "csv"

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.path)
        except OSError as exc:
            raise DataError(f"cannot read {self.path}: {exc}") from exc
        df.columns = [c.strip().lower() for c in df.columns]
        stamp = next((c for c in ("time", "date", "timestamp") if c in df.columns), None)
        missing = {"open", "high", "low", "close"} - set(df.columns)
        if stamp is None or missing:
            raise DataError(f"{self.path} needs time/date plus open,high,low,close (missing: {sorted(missing) or stamp})")
        numeric = pd.api.types.is_numeric_dtype(df[stamp])
        df.index = pd.to_datetime(df.pop(stamp), unit="s" if numeric else None, utc=True)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return _frame(df)

    def spot(self) -> float:
        return float(self._load()["close"].iloc[-1])

    def candles(self, days: int) -> pd.DataFrame:
        return self._load().tail(days)


def get_provider(name: str, csv_path: Path | None = None, product: str = "ETH-USD") -> Provider:
    """``product`` picks the market for Coinbase; Kraken here serves ETH-USD only, and a CSV is whatever it holds."""
    if name == "coinbase":
        return Coinbase(product)
    if name == "kraken":
        if product != "ETH-USD":
            raise DataError(f"the kraken provider serves ETH-USD only, not {product}")
        return Kraken()
    if name == "csv":
        if csv_path is None:
            raise DataError("the csv provider needs --csv PATH")
        return Csv(csv_path)
    raise DataError(f"unknown provider {name!r}")
