"""Price past the stop and the exchange stop gone (it fired, maybe only partly) -> the bot
closes the rest as a normal "stop", not "no_stop", and never says "half profit taken"
(SOL/BNB 2026-11-14: -2021 "would immediately trigger" when re-placing the stop)."""
from types import SimpleNamespace

from core import trader as trader_mod
from core.trader import Trader


def fake(price, has_stop=False):
    t = {"id": 1, "symbol": "SOL/USDT:USDT", "side": "short", "qty": 7.06, "partial_done": 0,
         "stop": 81.2203, "tp": 74.19}
    closed, updates, placed = [], [], []
    me = SimpleNamespace(
        open_trades=lambda: [t] if not closed else [],
        broker=SimpleNamespace(price=lambda s: price, has_stop=lambda s: has_stop,
                               place_stop=lambda *a: placed.append(a)),
        m=SimpleNamespace(is_open=lambda: True, mode="demo"),
        st=SimpleNamespace(update_trade=lambda *a, **k: updates.append(k), open_carries=lambda *a: []),
        acc=SimpleNamespace(name="amar", get=lambda k: None),
        close_trade=lambda tr, reason: closed.append(reason) or True,
        finish_trade=lambda *a: None, to=lambda: None,
        extend_tp=lambda *a: False)
    return me, closed, updates, placed


def test_stop_partly_filled_is_closed_as_stop(monkeypatch):
    msgs = []
    monkeypatch.setattr(trader_mod, "notify", lambda m, **k: msgs.append(m))
    me, closed, updates, placed = fake(81.45)
    Trader.reconcile(me, {"SOL/USDT:USDT": {"side": -1, "qty": 3.0, "entry": 79.34}})
    assert closed == ["stop"] and placed == [] and updates == []
    assert not any("HALF" in str(m) for m in msgs)


def test_missing_stop_below_stop_price_is_placed_again(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    me, closed, updates, placed = fake(80.0)
    Trader.reconcile(me, {"SOL/USDT:USDT": {"side": -1, "qty": 7.06, "entry": 79.34}})
    assert closed == [] and len(placed) == 1


def test_real_half_profit_still_reported(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    me, closed, updates, placed = fake(76.0, has_stop=True)
    Trader.reconcile(me, {"SOL/USDT:USDT": {"side": -1, "qty": 3.53, "entry": 79.34}})
    assert updates == [{"partial_done": 1}] and closed == []
