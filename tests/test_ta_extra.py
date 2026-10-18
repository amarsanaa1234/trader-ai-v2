"""Candidate chart methods: well-formed output and no peeking at future candles."""
import numpy as np
import pytest

from conftest import synthetic_frame
from core.ta_extra import CANDIDATES


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_values_and_no_lookahead(name):
    df = synthetic_frame(n=800, seed=9)
    full = CANDIDATES[name](df).to_numpy()
    assert set(np.unique(full)) <= {-1.0, 0.0, 1.0} and len(full) == len(df)
    assert np.abs(full).sum() > 0, "never fires"
    cut = 600   # the value at a candle must not change when later candles are added
    part = CANDIDATES[name](df.iloc[:cut]).to_numpy()
    assert np.array_equal(part, full[:cut])


def test_liquidity_sweep_rule():
    df = synthetic_frame(n=60, seed=2)
    i = 50
    prev_hi = df["high"].iloc[i - 20:i].max()
    df.iloc[i, df.columns.get_loc("high")] = prev_hi * 1.01       # pokes above the old high...
    df.iloc[i, df.columns.get_loc("close")] = prev_hi * 0.995     # ...but closes back below it
    df.iloc[i, df.columns.get_loc("low")] = min(df["low"].iloc[i], prev_hi * 0.99)
    assert CANDIDATES["sweep"](df).iloc[i] == -1.0
