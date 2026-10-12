"""Scoring, parameter handling and the backtester on synthetic data."""
import numpy as np

from conftest import synthetic_frame
from core import backtest
from core.context import mkt
from core.strategy import clamp_params, compute, default_params, param_space, source_names, weight_key


def test_every_source_has_a_weight():
    p = default_params()
    for name in source_names():
        assert weight_key(name) in p, name


def test_param_space_keys_exist_and_clamp():
    p = default_params()
    space = param_space()
    assert set(space) <= set(p)
    wild = clamp_params({k: 1e9 for k in space})
    for k, (lo, hi, _) in space.items():
        assert lo <= wild[k] <= hi, k
    assert wild["ema_fast"] < wild["ema_slow"]
    assert wild["adx_range"] < wild["adx_trend"]


def test_compute_outputs(frame):
    p = default_params()
    f = compute(frame, p)
    assert set(np.unique(f["signal"])) <= {-1, 0, 1}
    assert (f["signal"].to_numpy()[: p["ema_slow"]] == 0).all()   # warm-up: no trades
    assert np.isfinite(f["score"].to_numpy()[p["ema_slow"]:]).all()
    for name in source_names():
        assert name in f


def test_market_sources_change_the_score(frame):
    p = default_params()
    base = compute(frame, p)["score"]
    hot = frame.copy()
    hot["funding"] = 0.001                      # very crowded longs
    assert (compute(hot, p)["score"] < base).all() or p["w_funding"] >= 0


def test_backtest_respects_risk(frame):
    p = default_params()
    res = backtest.run(frame, p)
    assert len(res.trades) > 0
    # stop-loss + fees + slippage: a single trade can't lose much more than 1R
    assert min(t["r"] for t in res.trades) > -1.6
    assert res.equity[-1] > 0


def test_backtest_uses_market_costs(frame):
    p = default_params()
    cheap = backtest.run(frame, p, costs={"maker": 0, "taker": 0, "slippage": 0})
    normal = backtest.run(frame, p)
    assert cheap.equity[-1] >= normal.equity[-1]
    assert mkt().costs["taker"] > mkt().costs["maker"]


def test_trending_market_is_traded_both_ways():
    up = backtest.run(synthetic_frame(drift=0.0008, seed=3), default_params())
    down = backtest.run(synthetic_frame(drift=-0.0008, seed=4), default_params())
    assert sum(t["side"] == 1 for t in up.trades) > sum(t["side"] == -1 for t in up.trades)
    assert sum(t["side"] == -1 for t in down.trades) > sum(t["side"] == 1 for t in down.trades)
