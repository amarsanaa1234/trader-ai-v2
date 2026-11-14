"""The bot closes a trade itself once its take-profit is reached, in case the
exchange never got the take-profit order (Binance demo -2022, 2026-11-09)."""
from types import SimpleNamespace

from core import trader as trader_mod
from core.trader import Trader


def fake(price, market_open=True):
    t = {"id": 1, "symbol": "BNB/USDT:USDT", "side": "short", "qty": 1.14, "partial_done": 1,
         "stop": 699.53, "tp": 689.52}
    closed = []
    me = SimpleNamespace(
        open_trades=lambda: [t] if not closed else [],
        broker=SimpleNamespace(price=lambda s: price, has_stop=lambda s: True),
        m=SimpleNamespace(is_open=lambda: market_open, mode="demo"),
        st=SimpleNamespace(update_trade=lambda *a, **k: None, open_carries=lambda *a: []),
        acc=SimpleNamespace(name="amar", get=lambda k: None),
        close_trade=lambda tr, reason: closed.append(reason) or True,
        finish_trade=lambda *a: None, to=lambda: None,
        extend_tp=lambda *a: False)
    return me, closed


def test_short_past_its_tp_is_closed(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    me, closed = fake(685.17)
    Trader.reconcile(me, {"BNB/USDT:USDT": {"side": -1, "qty": 0.6, "entry": 719.78}})
    assert closed == ["tp"]


def test_not_yet_at_tp_or_market_closed_stays_open(monkeypatch):
    monkeypatch.setattr(trader_mod, "notify", lambda *a, **k: None)
    for price, is_open in ((695.0, True), (685.0, False)):
        me, closed = fake(price, is_open)
        Trader.reconcile(me, {"BNB/USDT:USDT": {"side": -1, "qty": 0.6, "entry": 719.78}})
        assert closed == []
