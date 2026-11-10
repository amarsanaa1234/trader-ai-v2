"""15m London breakout paper bot (core/orb15.py): no look-ahead, and brain 2 can run it."""
import numpy as np
import pandas as pd

from core import orb15
from core.storage import Storage

START = 1_790_000_000_000 // 86_400_000 * 86_400_000     # a midnight UTC


def candles(days=100, seed=3):
    rng = np.random.default_rng(seed)
    n = days * 96
    c = 150 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0006, n)) * c
    return pd.DataFrame({"ts": START + np.arange(n) * orb15.BAR_MS, "open": o, "high": np.maximum(o, c) + w,
                         "low": np.minimum(o, c) - w, "close": c})


def test_signals_never_use_later_candles():
    df = candles()
    full = orb15.signals(orb15.features(df))
    for cut in (3000, 5000, 7000):
        part = orb15.signals(orb15.features(df.iloc[:cut]))
        assert np.array_equal(full["dir"][:cut], part["dir"])
        assert np.allclose(full["sl"][:cut], part["sl"])


def test_trades_follow_the_rules():
    df = orb15.features(candles())
    trades, _ = orb15.simulate(df, orb15.signals(df), orb15.PAIRS["USD_JPY"])
    assert trades, "a random walk should break its London range now and then"
    t = pd.to_datetime([x["entry_ts"] for x in trades], unit="ms", utc=True).tz_convert("Europe/London")
    mins = t.hour * 60 + t.minute
    assert ((mins >= 8 * 60) & (mins <= 10 * 60)).all()                # entries only 08:00-10:00 London
    days = [x.date() for x in t]
    assert len(days) == len(set(days))                                  # at most one trade per pair per day
    assert all(x["r"] >= -1.2 for x in trades if x["reason"] == "stop")  # a stop costs about 1R


def test_brain2_step_saves_the_paper_account(tmp_path):
    st = Storage(tmp_path / "fx.db")
    for pair in orb15.PAIRS:
        df = candles(seed=hash(pair) % 100)
        st.db.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                          [(pair, "15m", int(r.ts), r.open, r.high, r.low, r.close, 0.0) for r in df.itertuples()])
    now = START + 100 * 86_400_000
    b = orb15.step(st, now)
    assert st.get(orb15.KEY)["n_trades"] == b["n_trades"] and b["start"] == now - 30 * 86_400_000
    assert b["live_trades"] == 0 and abs(b["return"]) < 1
