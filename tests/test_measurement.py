"""Honest measurement: shared-account backtest, currency exposure, random benchmark,
the demo stop rule, the shared Telegram inbox and the ensemble vote."""
import numpy as np
import pytest

from conftest import synthetic_frame
from core import live_check, portfolio, risk_guard as rg, significance
from core.strategy import compute, default_params


def test_currency_risk_adds_up_same_way_bets():
    legs = [("EUR_USD", 1, 0.01), ("GBP_USD", 1, 0.01)]          # two "USD falls" bets
    assert rg.currency_risk(legs, "AUD_USD", 1) == pytest.approx(0.02)   # a third one leans the same way
    assert rg.currency_risk(legs, "AUD_USD", -1) == 0.0               # the opposite bet is not added
    assert rg.currency_risk(legs, "USD_JPY", -1) == pytest.approx(0.02)  # selling USD_JPY = USD falls too
    assert rg.currency_risk([], "EUR_USD", 1) == 0.0


def test_portfolio_shares_one_account():
    frames = {f"S{k}": synthetic_frame(n=2500, drift=0.0004 * (k - 1), seed=k) for k in range(3)}
    p = {**default_params(), "entry_threshold": 0.8}
    res = portfolio.run(frames, p)
    assert res["trades"] > 0 and sum(res["per_symbol"].values()) == res["trades"]
    assert 0.0 <= res["max_dd"] < 1.0
    # one shared account cannot beat three separate accounts' position count limits
    alone = sum(len(__import__("core.backtest", fromlist=["run"]).run(df, p).trades) for df in frames.values())
    assert res["trades"] <= alone


def test_random_benchmark_shape():
    frames = {"A": synthetic_frame(n=2500, seed=3), "B": synthetic_frame(n=2500, seed=4)}
    p = {**default_params(), "entry_threshold": 0.8}
    out = significance.random_benchmark(frames, p, n=10)
    assert out["runs"] == 10 and 0.0 < out["p_value"] <= 1.0
    assert out["trades"] > 0


def test_live_check_stops_only_when_clearly_worse():
    good = [0.5, -0.4] * 20
    assert live_check.verdict(good, 0.1) is None
    assert live_check.verdict([-0.6, 0.2] * 10, 0.1) is None              # only 20 trades: too early
    bad = [-1.0, 0.3, -1.0, -0.2] * 10                                     # 40 trades, avg -0.475R
    assert "Paused" in live_check.verdict(bad, 0.1)
    assert live_check.verdict(bad, None) is None                           # nothing to compare with


def test_telegram_parse_targets_markets():
    from core.telegram_bot import parse
    assert parse("/stop", "fx") == "/stop"
    assert parse("/stop fx", "fx") == "/stop"
    assert parse("/stop crypto", "fx") is None
    assert parse("/Status@MyBot", "crypto") == "/status"


def test_telegram_hub_one_reader(tmp_path, monkeypatch):
    from core import telegram_bot as tg
    monkeypatch.setattr(tg, "HUB_DB", tmp_path / "tg.db")
    db = tg._hub()
    assert tg._take_lease(db, "crypto:1")
    assert not tg._take_lease(db, "fx:2")          # someone else holds it
    assert tg._take_lease(db, "crypto:1")          # the holder renews
    db.execute("UPDATE kv SET value=? WHERE key='poller'", ('{"owner": "crypto:1", "until": 0}',))
    assert tg._take_lease(db, "fx:2")              # expired -> taken over


def test_ensemble_needs_a_majority():
    df = synthetic_frame(n=1500, seed=5)
    a = {**default_params(), "entry_threshold": 0.6}
    b = {**default_params(), "entry_threshold": 3.0}     # almost never trades
    one = compute(df, a)["signal"].to_numpy()
    ens = compute(df, {**a, "ensemble": [a, b, b], "ensemble_votes": 2})["signal"].to_numpy()
    assert np.abs(ens).sum() < np.abs(one).sum()        # a lone voter can't trade on its own
    ens2 = compute(df, {**a, "ensemble": [a, a, b], "ensemble_votes": 2})["signal"].to_numpy()
    assert (ens2 == one).mean() > 0.95
