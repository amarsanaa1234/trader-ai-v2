"""Fixes from the 2026-11-09 audit (owner's trading skills)."""
import json
import time
from types import SimpleNamespace

import ccxt

from core import live_check, strategy
from core import trader as trader_mod
from core.context import mkt
from core.trader import Trader
from markets.crypto.broker import BinanceBroker


def test_edge_refresh_does_not_restart_the_live_check(tmp_path, monkeypatch):
    m = mkt()
    monkeypatch.setattr(type(m), "params_path", property(lambda self: tmp_path / "p.json"))
    monkeypatch.setattr(type(m), "params_history_dir", property(lambda self: tmp_path / "hist"))
    strategy.save_params(strategy.default_params(), {}, note="shadow promoted")
    first = live_check.since_settings_changed()
    time.sleep(0.01)
    strategy.save_params(strategy.default_params(), {}, note=strategy.EDGES_NOTE)
    assert live_check.since_settings_changed() == first          # same settings, same window
    strategy.save_params(strategy.default_params(), {}, note="owner change")
    assert live_check.since_settings_changed() >= first


def test_old_files_without_the_field_use_the_last_real_change(tmp_path, monkeypatch):
    m = mkt()
    monkeypatch.setattr(type(m), "params_path", property(lambda self: tmp_path / "p.json"))
    monkeypatch.setattr(type(m), "params_history_dir", property(lambda self: tmp_path / "hist"))
    (tmp_path / "hist").mkdir()
    (tmp_path / "hist" / "params_20261027_000000.json").write_text(
        json.dumps({"updated_utc": "20261026_091449", "note": "owner short-term rule"}))
    (tmp_path / "hist" / "params_20261108_000000.json").write_text(
        json.dumps({"updated_utc": "20261103_011843", "note": strategy.EDGES_NOTE}))
    (tmp_path / "p.json").write_text(json.dumps({"updated_utc": "20261109_012643", "note": strategy.EDGES_NOTE}))
    assert live_check.since_settings_changed() == 1793006089000       # 2026-10-26 09:14:49 UTC


def test_crypto_funding_and_fear_greed_weights_are_off():
    p = strategy.clamp_params({"w_funding": -0.95, "w_fng": -0.5, "w_funding_chg": 0.4})
    assert p["w_funding"] == p["w_fng"] == p["w_funding_chg"] == 0.0


def test_unknown_position_warns_only_when_it_stays(monkeypatch):
    sent = []
    monkeypatch.setattr(trader_mod, "notify", lambda msg, **k: sent.append(msg))
    me = SimpleNamespace(open_trades=lambda: [], m=SimpleNamespace(mode="demo"),
                         st=SimpleNamespace(open_carries=lambda *a: []),
                         acc=SimpleNamespace(name="amar", get=lambda k: None), to=lambda: None)
    Trader.reconcile(me, {"BNB/USDT:USDT": {"side": -1, "qty": 0.6}})
    assert sent == []                                  # a just-closed position can linger one loop
    Trader.reconcile(me, {"BNB/USDT:USDT": {"side": -1, "qty": 0.6}})
    assert len(sent) == 1 and "does not know" in sent[0]
    sent.clear()
    Trader.reconcile(me, {})
    Trader.reconcile(me, {"BNB/USDT:USDT": {"side": -1, "qty": 0.6}})
    assert sent == []


class TimeoutEx:
    def __init__(self, placed):
        self.placed, self.cid = placed, None

    def create_order(self, sym, typ, side, qty, px=None, params=None):
        self.cid = params["clientOrderId"]
        raise ccxt.RequestTimeout('binanceusdm {"code":-1007,"msg":"Timeout waiting for response"}')

    def fetch_order(self, oid, sym, params=None):
        if self.placed and params["origClientOrderId"] == self.cid:
            return {"id": "123", "status": "open"}
        raise ccxt.OrderNotFound("Order does not exist")


def test_entry_timeout_finds_the_order_by_our_id(monkeypatch):
    monkeypatch.setattr("markets.crypto.broker.time.sleep", lambda s: None)
    b = BinanceBroker.__new__(BinanceBroker)
    b.ex = TimeoutEx(placed=True)
    assert b._create_entry("BNB/USDT:USDT", "limit", "sell", 0.6, 700.0, {"postOnly": True})["id"] == "123"
    b.ex = TimeoutEx(placed=False)
    try:
        b._create_entry("BNB/USDT:USDT", "limit", "sell", 0.6, 700.0, {"postOnly": True})
        assert False, "should raise when the order really was not placed"
    except ccxt.RequestTimeout:
        pass
