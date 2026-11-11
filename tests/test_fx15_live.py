"""The 15m learner on OANDA (core/fx15_live.py): sizing caps, one position per pair, and the
learner's exits, against a fake broker."""
import pandas as pd

from core import fx15_live
from core import notifier
from core.storage import Storage


class FakeBroker:
    def __init__(self, nav=100_000.0):
        self.nav, self.pos, self.orders, self.closed = nav, {}, [], []

    def equity(self):
        return self.nav

    def positions(self):
        return dict(self.pos)

    def price(self, s):
        return 150.0 if s.endswith("JPY") else 1.3

    def check_size(self, s, qty, px):
        return qty, "ok"

    def _units(self, s, qty):
        return round(qty)

    def _px(self, s, x):
        return f"{x:.5f}"

    def _order(self, body):
        self.orders.append(body)
        s = body["instrument"]
        units = float(body["units"])
        self.pos[s] = {"side": 1 if units > 0 else -1, "qty": abs(units), "entry": self.price(s)}
        return {"orderFillTransaction": {"price": str(self.price(s))}}

    def close(self, s):
        self.pos.pop(s, None)
        self.closed.append(s)
        return self.price(s)

    def closed_result(self, s, trade):
        return self.price(s), -500.0


def decision(pair, session="london", side=1, dist=0.16, p=0.9):
    t0 = int(pd.Timestamp("2026-11-12 08:15", tz="Europe/London").tz_convert("UTC").timestamp() * 1000)
    return {"pair": pair, "arm": f"{pair}:{session}", "session": session, "side": side, "dist": dist,
            "weight": 1.0, "p_pos": p, "belief": 0.2, "entry_ts": t0}


def test_two_percent_risk_capped_by_leverage_and_one_per_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(fx15_live, "notify", lambda *a, **k: None)
    st, b = Storage(tmp_path / "fx.db"), FakeBroker()
    live = fx15_live.Live(st, b, None)
    d = decision("USD_JPY")
    live.enter([d, decision("USD_JPY", "ny"), decision("GBP_USD", dist=0.0016, p=0.8)], d["entry_ts"])
    assert [o["instrument"] for o in b.orders] == ["USD_JPY", "GBP_USD"]      # the second USD_JPY arm waits
    jpy = live.s["positions"]["USD_JPY"]
    assert jpy["qty"] * 150 <= 12 * b.nav + 1                                 # per-trade notional cap
    assert jpy["risk"] <= 0.02 * b.nav + 1
    total = sum(p["qty"] * p["entry"] for p in b.pos.values())
    assert total <= 25 * b.nav + 1                                            # total notional cap
    assert b.orders[0]["stopLossOnFill"]["distance"] == "0.16000"


def test_time_exit_and_stop_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(fx15_live, "notify", lambda *a, **k: None)
    st, b = Storage(tmp_path / "fx.db"), FakeBroker()
    live = fx15_live.Live(st, b, None)
    d = decision("USD_JPY")
    live.enter([d, decision("GBP_USD", dist=0.0016)], d["entry_ts"])
    b.pos.pop("GBP_USD")                                  # stopped out on OANDA
    before_end = int(pd.Timestamp("2026-11-12 15:30", tz="Europe/London").tz_convert("UTC").timestamp() * 1000)
    live.reconcile(before_end)
    assert [t["pair"] for t in live.s["trades"]] == ["GBP_USD"] and live.s["trades"][0]["reason"] == "stop"
    assert "USD_JPY" in live.s["positions"]
    at_end = int(pd.Timestamp("2026-11-12 16:00", tz="Europe/London").tz_convert("UTC").timestamp() * 1000)
    live.reconcile(at_end)
    assert b.closed == ["USD_JPY"] and live.s["trades"][-1]["reason"] == "session end"


def test_exit_rules():
    d = {"session": "tokyo", "entry_ts": int(pd.Timestamp("2026-11-12 09:15", tz="Asia/Tokyo").timestamp() * 1000)}
    at = lambda s, tz: int(pd.Timestamp(s, tz=tz).timestamp() * 1000)          # noqa: E731
    assert fx15_live.exit_due(d, at("2026-11-12 14:45", "Asia/Tokyo")) is None
    assert fx15_live.exit_due(d, at("2026-11-12 15:00", "Asia/Tokyo")) == "session end"
    fri = {"session": "ny", "entry_ts": at("2026-11-13 10:00", "America/New_York")}
    assert fx15_live.exit_due(fri, at("2026-11-13 20:00", "UTC")) == "Friday close"
