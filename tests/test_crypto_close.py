"""Closing on Binance must work even when the exchange misbehaves the way the
demo did on 2026-10-21 (BNB stayed open past its stop for two days)."""
import ccxt

from markets.crypto.broker import BinanceBroker


class FakeEx:
    def __init__(self, keep_orders_on_cancel_all=False, reject_reduce_only=False):
        self.orders = [{"id": "tp1"}]
        self.keep, self.reject = keep_orders_on_cancel_all, reject_reduce_only
        self.sent = []

    def fetch_positions(self, syms=None):
        return [{"symbol": "BNB/USDT:USDT", "contracts": 0.6, "side": "short"}]

    def cancel_all_orders(self, sym, params=None):
        if not self.keep:
            self.orders = []

    def fetch_open_orders(self, sym, params=None):
        return [] if params else list(self.orders)

    def cancel_order(self, oid, sym, params=None):
        self.orders = [o for o in self.orders if o["id"] != oid]

    def create_order(self, sym, typ, side, qty, px=None, params=None):
        reduce_only = bool((params or {}).get("reduceOnly"))
        if self.orders and reduce_only:
            raise ccxt.InvalidOrder('binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
        if self.reject and reduce_only:
            raise ccxt.InvalidOrder('binanceusdm {"code":-2022,"msg":"ReduceOnly Order is rejected."}')
        self.sent.append((side, qty, reduce_only))
        return {"average": 747.5}


def broker(ex):
    b = BinanceBroker.__new__(BinanceBroker)
    b.ex = ex
    b.positions = lambda: {"BNB/USDT:USDT": {"side": -1, "qty": 0.6, "entry": 735.0}}
    return b


def test_take_profit_left_by_cancel_all_is_removed_before_closing():
    ex = FakeEx(keep_orders_on_cancel_all=True)
    assert broker(ex).close("BNB/USDT:USDT") == 747.5
    assert ex.sent == [("buy", 0.6, True)] and not ex.orders


def test_rejected_reduce_only_falls_back_to_exact_opposite_order():
    ex = FakeEx(reject_reduce_only=True)
    assert broker(ex).close("BNB/USDT:USDT") == 747.5
    assert ex.sent == [("buy", 0.6, False)]
