"""Order execution for FX.

PaperBroker - "signal" mode: simulated fills on real OANDA bid/ask.
OandaBroker - demo (practice account) / live on OANDA v20.

Quantities: the core sizes positions so that qty * price_move = profit in the
ACCOUNT currency. OANDA trades in units of the base currency whose profit is in
the QUOTE currency, so units = qty / (quote -> account rate) on the way in and
qty = units * rate on the way out. Stop-loss and take-profit live on OANDA's
side (attached to the trade), so they work while the laptop sleeps.
"""
import logging
import os

from core.context import mkt
from core.storage import Storage, now_ms

from .oanda import Client, data_client

log = logging.getLogger(__name__)


def _quote(symbol: str) -> str:
    return symbol.split("_")[1]


def bid_ask(client: Client, symbols) -> dict:
    js = client.get(client.acct("/pricing"), instruments=",".join(symbols), includeHomeConversions="true")
    conv = {c["currency"]: float(c["positionValue"]) for c in js.get("homeConversions", [])}
    out = {}
    for p in js["prices"]:
        bid, ask = float(p["bids"][0]["price"]), float(p["asks"][0]["price"])
        out[p["instrument"]] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2,
                                "q2a": conv.get(_quote(p["instrument"]), 1.0), "tradeable": p.get("tradeable", True)}
    return out


class PaperBroker:
    """Same behaviour as the crypto paper broker, but fills at OANDA's real bid/ask
    (so the spread is paid like on a real account) and without carry."""

    def __init__(self, st: Storage, account):
        self.st, self.acc = st, account
        start = float(os.getenv("FX_PAPER_START_EQUITY") or mkt().cfg.get("paper_start_equity", 10000))
        self.s = account.get("paper") or {}
        self.s.setdefault("cash", start)
        for k in ("positions", "closed", "pending"):
            self.s.setdefault(k, {})

    def _save(self):
        self.acc.set("paper", self.s)

    def _q(self, symbol):
        return bid_ask(data_client(), [symbol])[symbol]

    def price(self, symbol):
        return self._q(symbol)["mid"]

    def check_size(self, symbol, qty, price):
        return (qty, "ok") if qty * price >= 100 else (0.0, "хэмжээ хэт бага")

    def place_entry(self, symbol, side, qty, limit):
        q = self._q(symbol)
        fill_side = q["ask"] if side == 1 else q["bid"]
        if mkt().entry_order == "market" or (limit - fill_side) * side >= 0:   # marketable: fills now
            self._fill_entry(symbol, side, qty, fill_side)
            return {"status": "filled", "filled": qty, "avg": fill_side}
        pend = {"id": f"paper-{now_ms()}", "side": side, "qty": qty, "limit": limit, "status": "open",
                "filled": 0.0, "avg": None}
        self.s["pending"][symbol] = pend
        self._save()
        return pend

    def _fill_entry(self, symbol, side, qty, px):
        self.s["positions"][symbol] = {"side": side, "qty": qty, "entry": px, "stop": None, "tp": None,
                                       "partial": None, "fees": 0.0, "realized": 0.0}
        self._save()

    def entry_status(self, symbol, pend):
        p = self.s["pending"].get(symbol)
        if not p:
            return {"status": pend.get("status", "canceled"), "filled": pend.get("filled", 0.0), "avg": pend.get("avg")}
        q = self._q(symbol)
        if (q["ask"] <= p["limit"]) if p["side"] == 1 else (q["bid"] >= p["limit"]):
            self._fill_entry(symbol, p["side"], p["qty"], p["limit"])
            self.s["pending"].pop(symbol)
            self._save()
            return {"status": "filled", "filled": p["qty"], "avg": p["limit"]}
        return {"status": "open", "filled": 0.0, "avg": None}

    def cancel_entry(self, symbol, pend):
        st = self.entry_status(symbol, pend)
        if st["status"] == "open":
            self.s["pending"].pop(symbol, None)
            self._save()
            return {"status": "canceled", "filled": 0.0, "avg": None}
        return st

    def protect(self, symbol, side, qty, stop, tp, partial_px):
        self.s["positions"][symbol].update(stop=stop, tp=tp, partial=partial_px)
        self._save()

    def move_stop(self, symbol, side, qty, new_stop):
        p = self.s["positions"].get(symbol)
        if p:
            q = self._q(symbol)
            exit_px = q["bid"] if side == 1 else q["ask"]
            if (exit_px <= new_stop) if side == 1 else (exit_px >= new_stop):
                raise ValueError("шинэ stop одоогийн үнийг давсан")
            p["stop"] = new_stop
            self._save()

    def has_stop(self, symbol):
        p = self.s["positions"].get(symbol)
        return bool(p and p["stop"])

    def place_stop(self, symbol, side, qty, stop):
        self.move_stop(symbol, side, qty, stop)

    def cancel_protection(self, symbol):
        pass

    def _close_qty(self, symbol, qty, px, final):
        p = self.s["positions"][symbol]
        gross = p["side"] * qty * (px - p["entry"])
        self.s["cash"] += gross
        p["realized"] += gross
        p["qty"] -= qty
        if final:
            self.s["positions"].pop(symbol)
            self.s["closed"][symbol] = {"exit": px, "pnl": p["realized"]}
        self._save()

    def positions(self):
        for sym, p in list(self.s["positions"].items()):
            if not p["stop"]:
                continue
            q, side = self._q(sym), p["side"]
            px = q["bid"] if side == 1 else q["ask"]          # what closing would get
            if (px <= p["stop"]) if side == 1 else (px >= p["stop"]):
                self._close_qty(sym, p["qty"], px, True)
                continue
            if p["partial"] and ((px >= p["partial"]) if side == 1 else (px <= p["partial"])):
                self._close_qty(sym, p["qty"] / 2, p["partial"], False)
                p["partial"] = None
                self._save()
            if p["tp"] and ((px >= p["tp"]) if side == 1 else (px <= p["tp"])):
                self._close_qty(sym, p["qty"], p["tp"], True)
        return {s: {"side": p["side"], "qty": p["qty"], "entry": p["entry"]} for s, p in self.s["positions"].items()}

    def equity(self):
        unreal = 0.0
        if self.s["positions"]:
            qs = bid_ask(data_client(), list(self.s["positions"]))
            for s, p in self.s["positions"].items():
                px = qs[s]["bid"] if p["side"] == 1 else qs[s]["ask"]
                unreal += p["side"] * p["qty"] * (px - p["entry"])
        return self.s["cash"] + unreal

    def close(self, symbol):
        p = self.s["positions"].get(symbol)
        if not p:
            return None
        q = self._q(symbol)
        px = q["bid"] if p["side"] == 1 else q["ask"]
        self._close_qty(symbol, p["qty"], px, True)
        return px

    def closed_result(self, symbol, trade):
        r = self.s["closed"].pop(symbol, None)
        self._save()
        return (r["exit"], r["pnl"]) if r else (None, None)


class OandaBroker:
    def __init__(self, st: Storage, account):
        token, acc_id = account.creds.get("oanda_token"), account.creds.get("oanda_account")
        if not token or not acc_id:
            raise RuntimeError(f"{account.name}: .env дотор USER<n>_FX_OANDA_TOKEN / _ACCOUNT алга байна")
        mode = mkt().mode
        if mode not in ("demo", "live"):
            raise RuntimeError(f"FX mode={mode}: demo эсвэл live байх ёстой")
        self.st, self.acc = st, account
        self.c = Client(token, acc_id, "live" if mode == "live" else "practice")
        self._prec = {}

    # ---- helpers ----
    def _precision(self, symbol):
        if symbol not in self._prec:
            i = self.c.get(self.c.acct("/instruments"), instruments=symbol)["instruments"][0]
            self._prec[symbol] = (int(i["displayPrecision"]), int(i["tradeUnitsPrecision"]),
                                  float(i["minimumTradeSize"]))
        return self._prec[symbol]

    def _px(self, symbol, px):
        return f"{px:.{self._precision(symbol)[0]}f}"

    def _units(self, symbol, qty, q2a=None):
        q2a = q2a or bid_ask(self.c, [symbol])[symbol]["q2a"]
        return round(qty / q2a, self._precision(symbol)[1])

    def _trades(self, symbol):
        return [t for t in self.c.get(self.c.acct("/openTrades"))["trades"] if t["instrument"] == symbol]

    def price(self, symbol):
        return bid_ask(self.c, [symbol])[symbol]["mid"]

    def equity(self):
        return float(self.c.get(self.c.acct("/summary"))["account"]["NAV"])

    def positions(self):
        out = {}
        pos = self.c.get(self.c.acct("/openPositions"))["positions"]
        if not pos:
            return out
        qs = bid_ask(self.c, [p["instrument"] for p in pos])
        for p in pos:
            for side, leg in ((1, p["long"]), (-1, p["short"])):
                units = abs(float(leg["units"]))
                if units > 0:
                    out[p["instrument"]] = {"side": side, "qty": units * qs[p["instrument"]]["q2a"],
                                            "entry": float(leg["averagePrice"])}
        return out

    def check_size(self, symbol, qty, price):
        units = self._units(symbol, qty)
        if units < self._precision(symbol)[2]:
            return 0.0, f"хэмжээ хэт бага ({units} unit)"
        return qty, "ok"

    # ---- entries ----
    def _order(self, body):
        return self.c.req("POST", self.c.acct("/orders"), json={"order": body})

    def place_entry(self, symbol, side, qty, limit):
        qty, why = self.check_size(symbol, qty, limit)
        if qty <= 0:
            raise ValueError(why)
        units = side * self._units(symbol, qty)
        if mkt().entry_order == "market":
            r = self._order({"type": "MARKET", "instrument": symbol, "units": str(units), "timeInForce": "FOK"})
            fill = r.get("orderFillTransaction")
            if not fill:
                raise ValueError(f"market захиалга биелсэнгүй: {r.get('orderCancelTransaction', {}).get('reason')}")
            return {"status": "filled", "filled": qty, "avg": float(fill["price"])}
        r = self._order({"type": "LIMIT", "instrument": symbol, "units": str(units),
                         "price": self._px(symbol, limit), "timeInForce": "GTC", "positionFill": "DEFAULT"})
        fill = r.get("orderFillTransaction")
        if fill:      # the limit was already marketable
            return {"status": "filled", "filled": qty, "avg": float(fill["price"])}
        return {"id": r["orderCreateTransaction"]["id"], "side": side, "qty": qty, "limit": limit,
                "status": "open", "filled": 0.0, "avg": None}

    def entry_status(self, symbol, pend):
        if pend.get("status") == "filled" and "id" not in pend:
            return pend
        o = self.c.get(self.c.acct(f"/orders/{pend['id']}"))["order"]
        if o["state"] == "FILLED":
            tx = self.c.get(self.c.acct(f"/transactions/{o['fillingTransactionID']}"))["transaction"]
            return {"status": "filled", "filled": pend["qty"], "avg": float(tx["price"])}
        return {"status": "open" if o["state"] == "PENDING" else "canceled", "filled": 0.0, "avg": None}

    def cancel_entry(self, symbol, pend):
        try:
            self.c.req("PUT", self.c.acct(f"/orders/{pend['id']}/cancel"))
        except Exception as e:
            log.debug("cancel entry %s: %s", symbol, e)
        st = self.entry_status(symbol, pend)
        return st if st["status"] == "filled" else {**st, "status": "canceled"}

    # ---- protection (attached to the trade on OANDA's side) ----
    def protect(self, symbol, side, qty, stop, tp, partial_px):
        trades = self._trades(symbol)
        if not trades:
            raise RuntimeError(f"{symbol}: нээлттэй trade олдсонгүй")
        for t in trades:     # stop first: if it fails the caller closes the position
            self.c.req("PUT", self.c.acct(f"/trades/{t['id']}/orders"),
                       json={"stopLoss": {"price": self._px(symbol, stop), "timeInForce": "GTC"}})
        try:
            for t in trades:
                if tp:
                    self.c.req("PUT", self.c.acct(f"/trades/{t['id']}/orders"),
                               json={"takeProfit": {"price": self._px(symbol, tp), "timeInForce": "GTC"}})
            if partial_px:
                half = round(abs(sum(float(t["currentUnits"]) for t in trades)) / 2, self._precision(symbol)[1])
                if half >= self._precision(symbol)[2]:
                    self._order({"type": "LIMIT", "instrument": symbol, "units": str(-side * half),
                                 "price": self._px(symbol, partial_px), "timeInForce": "GTC",
                                 "positionFill": "REDUCE_ONLY"})
        except Exception as e:
            log.warning("%s take-profit тавьж чадсангүй: %s", symbol, e)

    def has_stop(self, symbol):
        trades = self._trades(symbol)
        return bool(trades) and all(t.get("stopLossOrder") for t in trades)

    def place_stop(self, symbol, side, qty, stop):
        self.move_stop(symbol, side, qty, stop)

    def move_stop(self, symbol, side, qty, new_stop):
        # replacing the trade's stop is one atomic call: never without a stop
        for t in self._trades(symbol):
            self.c.req("PUT", self.c.acct(f"/trades/{t['id']}/orders"),
                       json={"stopLoss": {"price": self._px(symbol, new_stop), "timeInForce": "GTC"}})

    def cancel_protection(self, symbol):
        """Cancels leftover stand-alone orders (the partial take-profit). Orders attached
        to a trade disappear with it."""
        for o in self.c.get(self.c.acct("/pendingOrders"))["orders"]:
            if o.get("instrument") == symbol and o["type"] in ("LIMIT", "STOP", "MARKET_IF_TOUCHED"):
                try:
                    self.c.req("PUT", self.c.acct(f"/orders/{o['id']}/cancel"))
                except Exception as e:
                    log.debug("cancel %s: %s", o["id"], e)

    def close(self, symbol):
        p = self.positions().get(symbol)
        px = None
        if p:
            body = {"longUnits": "ALL"} if p["side"] == 1 else {"shortUnits": "ALL"}
            r = self.c.req("PUT", self.c.acct(f"/positions/{symbol}/close"), json=body)
            fill = r.get("longOrderFillTransaction") or r.get("shortOrderFillTransaction") or {}
            px = float(fill.get("price") or self.price(symbol))
        self.cancel_protection(symbol)
        return px

    def closed_result(self, symbol, trade):
        since = (int(trade["opened_ts"]) - 60_000) / 1000
        closed = [t for t in self.c.get(self.c.acct("/trades"), state="CLOSED", instrument=symbol, count=50)["trades"]
                  if float(t["openTime"]) >= since]
        if not closed:
            return None, None
        pnl = sum(float(t.get("realizedPL", 0)) + float(t.get("financing", 0)) for t in closed)
        exit_px = float(closed[0].get("averageClosePrice") or 0) or None
        return exit_px, pnl


def make_broker(st: Storage, account):
    return PaperBroker(st, account) if mkt().mode == "signal" else OandaBroker(st, account)
