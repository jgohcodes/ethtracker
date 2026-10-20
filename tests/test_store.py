import pandas as pd
import pytest

from ethtracker import data
from ethtracker.store import CachedProvider, merge


class Fake:
    """Stands in for a network provider: `days` daily candles ending `lag` days ago, close = self.close."""

    name = "fake"

    def __init__(self, lag: int = 0):
        self.calls: list[int] = []
        self.fail = False
        self.close = 100.0
        self.lag = lag

    def spot(self) -> float:
        return 1.0

    def candles(self, days: int) -> pd.DataFrame:
        self.calls.append(days)
        if self.fail:
            raise data.DataError("offline")
        end = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=self.lag)
        idx = pd.date_range(end=end, periods=days, freq="D")
        return pd.DataFrame(
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": self.close, "volume": 10.0}, index=idx
        )


@pytest.fixture
def fake():
    return Fake()


@pytest.fixture
def cached(fake, tmp_path):
    return CachedProvider(fake, tmp_path / "candles")


def test_first_call_downloads_everything_and_writes_the_cache(fake, cached):
    df = cached.candles(60)
    assert fake.calls == [60] and len(df) == 60
    assert cached.path.name == "fake-ETH-USD-1d.csv" and cached.path.exists()
    assert cached.spot() == 1.0 and cached.name == "fake"


def test_later_calls_fetch_only_the_gap_and_refresh_the_latest_candle(fake, cached):
    cached.candles(60)
    fake.close = 111.0
    df = cached.candles(60)
    assert fake.calls == [60, 3]  # zero missing days + 3 days of padding
    assert len(df) == 60 and df.index.is_unique and df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 111.0 and df["close"].iloc[0] == 100.0


def test_a_stale_cache_is_topped_up_with_exactly_the_missing_days(tmp_path):
    old = Fake(lag=10)
    CachedProvider(old, tmp_path / "candles").candles(60)
    fresh = Fake()
    df = CachedProvider(fresh, tmp_path / "candles").candles(60)
    assert fresh.calls == [13]  # 10 missing days + 3 padding
    assert df.index[-1] == pd.Timestamp.now(tz="UTC").normalize()
    assert (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_asking_for_more_history_than_is_cached_downloads_again(fake, cached):
    cached.candles(30)
    df = cached.candles(90)
    assert fake.calls == [30, 90] and len(df) == 90


def test_a_failed_refresh_falls_back_to_the_cache_with_a_warning(fake, cached, capsys):
    cached.candles(30)
    fake.fail = True
    df = cached.candles(30)
    assert len(df) == 30
    assert "using cached candles" in capsys.readouterr().err


def test_a_failed_download_with_no_cache_raises(fake, cached):
    fake.fail = True
    with pytest.raises(data.DataError, match="offline"):
        cached.candles(30)


def test_a_corrupt_cache_is_ignored_and_rebuilt(fake, cached):
    cached.candles(30)
    cached.path.write_text("not,a,candle,file\n1,2")
    assert len(cached.candles(30)) == 30
    assert fake.calls == [30, 30]


def test_merge_prefers_the_newer_candle_on_overlap():
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    old = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)
    new = pd.DataFrame({"close": [9.0, 4.0]}, index=idx[2:].append(idx[2:] + pd.Timedelta(days=1)))
    merged = merge(old, new)
    assert list(merged["close"]) == [1.0, 2.0, 9.0, 4.0]
