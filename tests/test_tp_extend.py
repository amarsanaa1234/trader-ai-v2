"""Fixed TP-extension rule (owner, 2026-11-15): within 0.1R of the TP and the score still at
least half the entry threshold the trade's way -> stop 0.15R behind the TP, TP +1R."""
from types import SimpleNamespace

from core import exits
from core import trader as trader_mod
from core.trader import Trader


def test_rule_numbers():
    # short, entry 100, 1R = 10, TP 72.6 (2.74R), threshold 1.5
    assert exits.tp_extension(-1, 72.6, 110, 10, 74.0, -2.0, 1.5) is None          # 0.14R away: not yet
    tp, stop = exits.tp_extension(-1, 72.6, 110, 10, 73.5, -0.75, 1.5)            # 0.09R away, score = half
    assert round(tp, 6) == 62.6 and round(stop, 6) == 74.1
    assert exits.tp_extension(-1, 72.6, 110, 10, 73.5, -0.74, 1.5) is None         # below half the threshold
    assert exits.tp_extension(-1, 72.6, 110, 10, 73.5, +2.0, 1.5) is None          # score turned the other way
    assert exits.tp_extension(-1, 72.6, 110, 10, 73.5, None, 1.5) is None
    tp, stop = exits.tp_extension(1, 127.4, 100, 10, 127.0, 3.0, 1.5)              # long; never loosens the stop
    assert round(tp, 6) == 137.4 and round(stop, 6) == 125.9
    assert exits.tp_extension(1, 127.4, 126.5, 10, 127.0, 3.0, 1.5)[1] == 126.5


def fake(price, score, tp_extend=True):
    t = {"id": 1, "symbol": "SOL/USDT:USDT", "side": "short", "qty": 2.0, "partial_done": 1, "entry": 100.0,
         "stop": 101.0, "tp": 72.6, "dist": 10.0, "risk_usd": 20.0}
    calls, updates = [], []
    me = SimpleNamespace(
        m=SimpleNamespace(cfg={"tp_extend": tp_extend}),
        st=SimpleNamespace(get=lambda k: score, update_trade=lambda i, **k: updates.append(k)),
        broker=SimpleNamespace(move_stop=lambda *a: calls.append(("stop",) + a),
                               move_tp=lambda *a: calls.append(("tp",) + a)),
        params={"entry_threshold": 1.5}, acc=SimpleNamespace(name="amar"), to=lambda: None)
    return me, t, calls, updates


def test_live_extends(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    me, t, calls, updates = fake(73.4, -1.2)
    assert Trader.extend_tp(me, t, -1, 73.4, {"SOL/USDT:USDT": {"qty": 1.0}})
    assert [c[0] for c in calls] == ["stop", "tp"] and round(calls[0][4], 6) == 74.1 and round(calls[1][4], 6) == 62.6
    assert round(updates[0]["tp"], 6) == 62.6 and round(updates[0]["stop"], 6) == 74.1


def test_live_no_extend(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    for price, score, on in ((73.4, -0.5, True), (75.0, -3.0, True), (73.4, -3.0, False)):
        me, t, calls, updates = fake(price, score, on)
        assert not Trader.extend_tp(me, t, -1, price, {"SOL/USDT:USDT": {"qty": 1.0}})
        assert calls == [] and updates == []


def test_only_crypto_and_commodities():
    import markets
    assert markets.get("crypto").cfg.get("tp_extend") and markets.get("commodities").cfg.get("tp_extend")
    assert not markets.get("fx").cfg.get("tp_extend")
