"""FX fixes from the 2026-10-25 audit: limit waits 2 candles, fills need the spread traded
through, 401 is retried, weak live settings are flagged."""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core import backtest, improver
from core.storage import Storage
from core.strategy import default_params

COSTS = {"maker": 0.0002, "taker": 0.0005, "slippage": 0.0}


def frame(n=60, price=100.0):
    idx = pd.Index(1_700_000_000_000 + np.arange(n) * 3_600_000, name="ts")
    df = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1.0}, index=idx)
    df["hold_cost"] = 0.0
    return df


def test_limit_may_wait_two_candles(monkeypatch):
    df = frame()
    df.iloc[11, df.columns.get_loc("low")] = 99.5      # candle 1 after the signal: limit 99 not reached
    df.iloc[12, df.columns.get_loc("low")] = 98.5      # candle 2: reached
    f = pd.DataFrame({"signal": 0, "atr": 2.0}, index=df.index)
    f.iloc[10, 0] = 1
    p = {**default_params(), "entry_offset_atr": 0.5, "max_hold_bars": 1000}
    one = backtest.run(df, p, feats=f, entry_order="maker", costs=COSTS)
    assert one.missed == 1                                   # crypto/commodities: one candle
    monkeypatch.setattr(backtest, "entry_wait_candles", lambda: 2)
    two = backtest.run(df, p, feats=f, entry_order="maker", costs=COSTS)
    assert two.missed == 0                                   # FX: still waiting on candle 2 -> filled


def test_fill_needs_the_spread_traded_through():
    df = frame()                                             # low = 99 = limit: a touch
    f = pd.DataFrame({"signal": 0, "atr": 2.0}, index=df.index)
    f.iloc[10, 0] = 1
    p = {**default_params(), "entry_offset_atr": 0.5, "max_hold_bars": 1000}
    assert backtest.run(df, p, feats=f, entry_order="maker", costs=COSTS).missed == 0
    assert backtest.run(df, p, feats=f, entry_order="maker", costs={**COSTS, "fill_through": 0.0001}).missed == 1


def test_oanda_401_is_asked_again(monkeypatch):
    from markets.fx import oanda
    c = oanda.Client("token", "acc")
    answers = iter([SimpleNamespace(status_code=401, ok=False, text="no"),
                    SimpleNamespace(status_code=200, ok=True, text="", json=lambda: {"ok": 1})])
    monkeypatch.setattr(c.s, "request", lambda *a, **k: next(answers))
    monkeypatch.setattr(oanda.time, "sleep", lambda s: None)
    assert c.get("/x") == {"ok": 1}


def test_weak_settings_are_flagged_and_cleared():
    st = Storage(os.path.join(tempfile.mkdtemp(), "t.db"))
    h = improver.settings_health(st, {"positive_folds": 1}, {"return": -0.01})
    assert h["weak"] and st.get("settings_health")["weak"]
    assert not improver.settings_health(st, {"positive_folds": 3}, {"return": -0.01})["weak"]
    assert not improver.settings_health(st, {"positive_folds": 1}, {"return": 0.01})["weak"]
