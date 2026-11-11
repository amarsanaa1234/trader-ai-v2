"""The 15m learner (core/orb_learn.py) learns only from trades closed before a decision,
trusts an arm that keeps winning, and stops trading a situation that keeps losing."""
import json

import numpy as np

from core import orb_learn
from tests.test_orb15 import START, candles

DAY = 86_400_000


def trade(k, arm, r, width=1.2, day=0):
    t0 = START + (day if day else k) * DAY
    return {"arm": arm, "pair": arm.split(":")[0], "session": arm.split(":")[1], "entry_ts": t0,
            "exit_ts": t0 + 6 * 3_600_000, "r": r, "side": 1, "width": width, "trend": 0.5, "vol": 1.0, "dow": 2}


def test_decisions_never_use_trades_that_closed_later():
    rng = np.random.default_rng(1)
    cands = [trade(k, f"P{k % 3}:london", float(rng.normal(0.3, 1.0))) for k in range(300)]
    full = orb_learn.replay(cands, START)["book"]
    part = orb_learn.replay(cands[:150], START)["book"]
    assert [(t["weight"], t["p_pos"]) for t in full[:150]] == [(t["weight"], t["p_pos"]) for t in part]


def test_a_winning_arm_earns_risk_and_a_losing_one_stays_on_paper():
    cands = [trade(k, "GOOD:london" if k % 2 else "BAD:tokyo", 0.6 if k % 2 else -0.6) for k in range(400)]
    book = orb_learn.replay(cands, START)["book"]
    good = [t["weight"] for t in book if t["arm"] == "GOOD:london"]
    bad = [t["weight"] for t in book if t["arm"] == "BAD:tokyo"]
    assert good[0] == 0 and good[-1] == 1.0          # starts on paper, earns its risk
    assert max(bad) == 0                              # never trusted


def test_lesson_from_a_situation_that_keeps_losing():
    rng = np.random.default_rng(2)
    cands = []
    for k in range(600):
        narrow = k % 3 == 0
        cands.append(trade(k, f"P{k % 4}:london", float(rng.normal(-0.8 if narrow else 0.3, 0.8)),
                           width=0.5 if narrow else 1.2))
    r = orb_learn.replay(cands, START)
    assert "width=low" in r["rules"]
    late = [t for t in r["book"][-100:] if t["width"] < 0.8]
    assert late and all("width=low" in t["blocked_by"] and t["weight"] == 0 for t in late)


def test_brain2_step_carries_on_from_the_snapshot(tmp_path, monkeypatch):
    from core.storage import Storage
    st = Storage(tmp_path / "fx.db")
    for pair in orb_learn.PAIRS:
        df = candles(seed=hash(pair) % 100)
        st.db.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)",
                          [(pair, "15m", int(r.ts), r.open, r.high, r.low, r.close, 0.0) for r in df.itertuples()])
    snap = tmp_path / "snap.json"
    end = START + 60 * DAY
    snap.write_text(json.dumps({"end": end, "trades": [trade(k, "USD_JPY:london", 0.4) for k in range(50)]}))
    monkeypatch.setattr(orb_learn, "snapshot_path", lambda: snap)
    from core import ai
    monkeypatch.setattr(ai, "enabled", lambda: False)         # no AI on a test machine
    b = orb_learn.step(st, START + 100 * DAY)
    assert b["start"] == end and b["learned_from"] >= 50
    assert all(t["ts"] > end for t in b["trades"])
