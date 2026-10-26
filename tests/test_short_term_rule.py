"""Owner rule (2026-10-26): short-term trading only, whatever the self-improver finds."""
import random

import pytest

import markets
from core import improver
from core.context import activate, mkt
from core.strategy import MAX_HOLD_DAYS, clamp_params, default_params


@pytest.mark.parametrize("name", markets.AVAILABLE)
def test_threshold_fixed_and_hold_capped(name, monkeypatch):
    prev = mkt()
    activate(markets.get(name))
    try:
        fixed = {"crypto": 1.5, "fx": 2.0, "commodities": 2.0}[name]
        p = clamp_params({**default_params(), "entry_threshold": 2.3, "max_hold_bars": 168})
        assert p["entry_threshold"] == fixed
        assert p["max_hold_bars"] * mkt().tf_ms() <= MAX_HOLD_DAYS * 86_400_000
        rng = random.Random(1)
        for _ in range(50):      # the improver's own candidates obey it too
            c = improver.mutate(p, 0.3, rng)
            assert c["entry_threshold"] == fixed
            assert c["max_hold_bars"] * mkt().tf_ms() <= MAX_HOLD_DAYS * 86_400_000
            r = improver.random_params(rng, p)
            assert r["entry_threshold"] == fixed
    finally:
        activate(prev)


def test_daily_candles_are_refused(monkeypatch):
    monkeypatch.setenv("CRYPTO_TIMEFRAME", "1d")
    with pytest.raises(ValueError, match="short-term"):
        _ = markets.get("crypto").timeframe
    monkeypatch.setenv("CRYPTO_TIMEFRAME", "4h")
    assert markets.get("crypto").timeframe == "4h"
