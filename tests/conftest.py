import numpy as np
import pandas as pd
import pytest


def synthetic_candles(n: int = 500, seed: int = 7) -> pd.DataFrame:
    """Seeded random walk. This is NOT market data; it only exercises the code paths."""
    rng = np.random.default_rng(seed)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    open_ = np.r_[2000.0, close[:-1]]
    spread = np.abs(rng.normal(0, 0.015, n))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": 1000.0,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC"),
    )


@pytest.fixture
def candles() -> pd.DataFrame:
    return synthetic_candles()


@pytest.fixture
def candles_csv(tmp_path, candles):
    path = tmp_path / "eth.csv"
    out = candles.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHTRACKER_HOME", str(tmp_path / "home"))
