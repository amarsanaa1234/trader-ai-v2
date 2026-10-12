"""Test set-up: activate the crypto market and build synthetic candles, so the
tests never touch the network or the real data folder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import markets  # noqa: E402
from core.context import activate  # noqa: E402

activate(markets.get("crypto"))


def synthetic_frame(n: int = 3000, drift: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Random-walk hourly candles with every column the crypto market expects."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.004, n)) * close
    idx = pd.Index(1_700_000_000_000 + np.arange(n) * 3_600_000, name="ts")
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + spread,
                       "low": np.minimum(open_, close) - spread, "close": close,
                       "volume": rng.uniform(100, 1000, n)}, index=idx)
    df["hold_cost"] = np.where(np.arange(n) % 8 == 0, 0.0001, 0.0)
    df["funding"] = 0.0001
    df["fng"] = 50.0
    for col in ("global_ls", "top_ls", "oi_value"):
        df[col] = np.nan
    df["news"] = 0.0
    return df


@pytest.fixture
def frame():
    return synthetic_frame()
