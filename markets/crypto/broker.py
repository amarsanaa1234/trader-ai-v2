"""Order execution.

PaperBroker   - "signal" mode: no API keys, simulated fills on real prices.
BinanceBroker - demo / testnet / live on Binance USDⓈ-M futures (+ spot for
                the funding carry).

Both work the same way:
  place_entry()  post-only limit (maker) entry, or market if the market's entry_order=market
  entry_status() / cancel_entry()
  protect()      stop-loss + partial take-profit + final take-profit, all ON
                 THE EXCHANGE, so they work if the laptop sleeps
  move_stop()    breakeven / trailing: new stop first, then cancel the old one
"""
import logging
import os

import ccxt

from core import risk_guard
from core.context import mkt
from core.storage import Storage, now_ms

from .data import public_exchange

log = logging.getLogger(__name__)
TAKER, MAKER, SPOT_FEE = 0.0005, 0.0002, 0.001
PAPER_SLIPPAGE = 0.0003


def _base(sym: str) -> str:
    return sym.split("/")[0]


class PaperBroker:
    def __init__(self, st: Storage, account):
        self.st, self.acc = st, account
        start = float(os.getenv("PAPER_START_EQUITY") or mkt().cfg.get("paper_start_equity", 1000))
        self.s = account.get("paper") or {}
        self.s.setdefault("cash", start)
        for k in ("positions", "closed", "pending", "carry"):
            self.s.setdefault(k, {})

    def _save(self):
        self.acc.set("paper", self.s)

    def price(self, symbol):
        return float(public_exchange().fetch_ticker(symbol)["last"])

    def check_size(self, symbol, qty, price):
        return (qty, "ok") if qty * price >= 5 else (0.0, "хэмжээ хэт бага")

    # ---- entries ----
    def place_entry(self, symbol, side, qty, limit):
        px = self.price(symbol)
        if mkt().entry_order == "market":
            fill = px * (1 + PAPER_SLIPPAGE * side)
            self._fill_entry(symbol, side, qty, fill, TAKER)
            return {"status": "filled", "filled": qty, "avg": fill}
        limit = min(limit, px) if side == 1 else max(limit, px)  # post-only can't cross
        pend = {"id": f"paper-{now_ms()}", "side": side, "qty": qty, "limit": limit, "status": "open",
                "filled": 0.0, "avg": None}
        self.s["pending"][symbol] = pend
        self._save()
        return pend

    def _fill_entry(self, symbol, side, qty, px, fee_rate):
        fee = qty * px * fee_rate
        self.s["cash"] -= fee
        self.s["positions"][symbol] = {"side": side, "qty": qty, "entry": px, "stop": None, "tp": None,
                                       "partial": None, "fees": fee, "realized": 0.0, "orig_qty": qty}
        self._save()

    def entry_status(self, symbol, pend):
        p = self.s["pending"].get(symbol)
        if not p:
            return {"status": pend.get("status", "canceled"), "filled": pend.get("filled", 0.0), "avg": pend.get("avg")}
        px = self.price(symbol)
        if (px <= p["limit"]) if p["side"] == 1 else (px >= p["limit"]):
            self._fill_entry(symbol, p["side"], p["qty"], p["limit"], MAKER)
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

    # ---- protection ----
    def protect(self, symbol, side, qty, stop, tp, partial_px):
        p = self.s["positions"][symbol]
        p.update(stop=stop, tp=tp, partial=partial_px)
        self._save()

    def move_stop(self, symbol, side, qty, new_stop):
        p = self.s["positions"].get(symbol)
        if p:
            px = self.price(symbol)
            if (px <= new_stop) if side == 1 else (px >= new_stop):
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

    # ---- positions ----
    def _close_qty(self, symbol, qty, px, fee_rate, final):
        p = self.s["positions"][symbol]
        gross = p["side"] * qty * (px - p["entry"])
        fee = qty * px * fee_rate
        self.s["cash"] += gross - fee
        p["realized"] += gross - fee
        p["qty"] -= qty
        if final:
            self.s["positions"].pop(symbol)
            self.s["closed"][symbol] = {"exit": px, "pnl": p["realized"] - p["fees"]}
        self._save()

    def positions(self):
        # simulate the exchange's resting stop / partial / take-profit orders
        for sym, p in list(self.s["positions"].items()):
            if not p["stop"]:
                continue
            px, side = self.price(sym), p["side"]
            if (px <= p["stop"]) if side == 1 else (px >= p["stop"]):
                self._close_qty(sym, p["qty"], px * (1 - PAPER_SLIPPAGE * side), TAKER, True)
                continue
            if p["partial"] and ((px >= p["partial"]) if side == 1 else (px <= p["partial"])):
                self._close_qty(sym, p["qty"] / 2, p["partial"], MAKER, False)
                p["partial"] = None
                self._save()
            if p["tp"] and ((px >= p["tp"]) if side == 1 else (px <= p["tp"])):
                self._close_qty(sym, p["qty"], p["tp"], MAKER, True)
        return {s: {"side": p["side"], "qty": p["qty"], "entry": p["entry"]} for s, p in self.s["positions"].items()}

    def equity(self):
        unreal = sum(p["side"] * p["qty"] * (self.price(s) - p["entry"]) for s, p in self.s["positions"].items())
        carry = sum(c["qty"] * ((px := self.price(s)) - c["spot_px"]) + c["qty"] * (c["perp_px"] - px)
                    for s, c in self.s["carry"].items())
        return self.s["cash"] + unreal + carry

    def close(self, symbol):
        p = self.s["positions"].get(symbol)
        if not p:
            return None
        px = self.price(symbol) * (1 - PAPER_SLIPPAGE * p["side"])
        self._close_qty(symbol, p["qty"], px, TAKER, True)
        return px

    def closed_result(self, symbol, trade):
        r = self.s["closed"].pop(symbol, None)
        self._save()
        return (r["exit"], r["pnl"]) if r else (None, None)

    # ---- funding carry (spot long + perp short) ----
    def carry_open(self, symbol, notional):
        px = self.price(symbol)
        qty = notional / px
        fees = qty * px * (SPOT_FEE + TAKER)
        self.s["cash"] -= fees
        self.s["carry"][symbol] = {"qty": qty, "spot_px": px, "perp_px": px, "since": now_ms()}
        self._save()
        return {"qty": qty, "spot_px": px, "perp_px": px, "fees": fees}

    def carry_close(self, symbol, qty):
        c = self.s["carry"].pop(symbol, None)
        if not c:
            return None
        px = self.price(symbol)
        fees = c["qty"] * px * (SPOT_FEE + TAKER)
        self.s["cash"] += c["qty"] * (px - c["spot_px"]) + c["qty"] * (c["perp_px"] - px) - fees
        self._save()
        return {"spot_px": px, "perp_px": px, "fees": fees}

    def carry_legs(self, symbol):
        c = self.s["carry"].get(symbol)
        return {"spot_qty": c["qty"] if c else 0.0, "perp_qty": c["qty"] if c else 0.0}

    def carry_funding(self, symbol, since_ms):
        """Credits funding paid since `since_ms` to the short leg; returns (amount, last_ts)."""
        c = self.s["carry"].get(symbol)
        if not c:
            return 0.0, since_ms
        rows = public_exchange().fetch_funding_rate_history(symbol, since=since_ms + 1, limit=100)
        total, last = 0.0, since_ms
        for r in rows:
            mark = float(r.get("info", {}).get("markPrice") or 0) or self.price(symbol)
            total += c["qty"] * mark * r["fundingRate"]   # shorts receive positive funding
            last = max(last, r["timestamp"])
        self.s["cash"] += total
        self._save()
        return total, last


class BinanceBroker:
    def __init__(self, st: Storage, account):
        API_KEY, API_SECRET, MODE = account.creds.get("binance_key"), account.creds.get("binance_secret"), mkt().mode
        if not API_KEY or not API_SECRET:
            raise RuntimeError(f"{account.name}: .env дотор USER<n>_CRYPTO_BINANCE_KEY / _SECRET алга байна")
        self.st, self.acc = st, account
        opts = {"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True}
        ex = ccxt.binanceusdm({**opts, "options": {"disableFuturesSandboxWarning": True}})
        spot = ccxt.binance(opts)
        if MODE == "demo":
            ex.enable_demo_trading(True)
            spot.enable_demo_trading(True)
        elif MODE == "testnet":
            ex.set_sandbox_mode(True)
            spot = None  # the old futures testnet has no matching spot account
        elif MODE != "live":
            raise RuntimeError(f"Үл мэдэгдэх crypto mode={MODE}")
        ex.load_markets()
        if spot:
            spot.load_markets()
        self.ex, self.spot = ex, spot
        self._ready = set()

    def _setup(self, symbol, leverage=risk_guard.MAX_LEVERAGE):
        if (symbol, leverage) in self._ready:
            return
        for fn in (lambda: self.ex.set_margin_mode("isolated", symbol),
                   lambda: self.ex.set_leverage(leverage, symbol)):
            try:
                fn()
            except Exception as e:  # "no need to change" errors are fine
                log.debug("setup %s: %s", symbol, e)
        self._ready.add((symbol, leverage))

    def _px(self, symbol, px):
        return float(self.ex.price_to_precision(symbol, px))

    def price(self, symbol):
        return float(self.ex.fetch_ticker(symbol)["last"])

    def equity(self):
        bal = self.ex.fetch_balance()
        v = bal.get("info", {}).get("totalMarginBalance")
        total = float(v) if v is not None else float(bal["total"].get("USDT", 0))
        # carry spot legs live in the spot wallet; count them so the kill switch sees the hedge
        for c in self.st.open_carries(mkt().mode, self.acc.name):
            total += c["qty"] * self.price(c["symbol"])
        return total

    def positions(self):
        out = {}
        for p in self.ex.fetch_positions():
            qty = float(p.get("contracts") or 0)
            if qty > 0:
                out[p["symbol"]] = {"side": 1 if p["side"] == "long" else -1, "qty": qty,
                                    "entry": float(p["entryPrice"])}
        return out

    def check_size(self, symbol, qty, price):
        m = self.ex.market(symbol)
        qty = float(self.ex.amount_to_precision(symbol, qty))
        min_amt = m["limits"]["amount"]["min"] or 0
        min_cost = m["limits"]["cost"]["min"] or 0
        if qty < min_amt or qty * price < min_cost:
            return 0.0, f"хэмжээ хэт бага (qty={qty}, notional≈{qty * price:.1f} USDT, доод хязгаар {min_cost})"
        return qty, "ok"

    # ---- entries ----
    def place_entry(self, symbol, side, qty, limit):
        self._setup(symbol)
        book = self.ex.fetch_order_book(symbol, 5)
        best = book["bids"][0][0] if side == 1 else book["asks"][0][0]
        px = min(limit, best) if side == 1 else max(limit, best)
        qty, why = self.check_size(symbol, qty, px)
        if qty <= 0:
            raise ValueError(why)
        order_side = "buy" if side == 1 else "sell"
        if mkt().entry_order == "market":
            o = self.ex.create_order(symbol, "market", order_side, qty)
            return {"status": "filled", "filled": qty, "avg": float(o.get("average") or self.price(symbol))}
        o = self.ex.create_order(symbol, "limit", order_side, qty, self._px(symbol, px), {"postOnly": True})
        return {"id": o["id"], "side": side, "qty": qty, "limit": px, "status": "open", "filled": 0.0, "avg": None}

    def entry_status(self, symbol, pend):
        if pend.get("status") == "filled" and "id" not in pend:
            return pend
        o = self.ex.fetch_order(pend["id"], symbol)
        filled = float(o.get("filled") or 0)
        avg = float(o.get("average") or o.get("price") or 0) or None
        status = {"closed": "filled", "open": "open"}.get(o["status"], "canceled")
        return {"status": status, "filled": filled, "avg": avg}

    def cancel_entry(self, symbol, pend):
        try:
            self.ex.cancel_order(pend["id"], symbol)
        except Exception as e:
            log.debug("cancel entry %s: %s", symbol, e)
        st = self.entry_status(symbol, pend)
        return st if st["status"] == "filled" else {**st, "status": "canceled"}

    # ---- protection ----
    def _stop_order(self, symbol, side, qty, stop):
        return self.ex.create_order(symbol, "market", "sell" if side == 1 else "buy", qty, None,
                                    {"reduceOnly": True, "workingType": "MARK_PRICE",
                                     "stopLossPrice": self._px(symbol, stop)})

    def protect(self, symbol, side, qty, stop, tp, partial_px):
        self._stop_order(symbol, side, qty, stop)  # if this raises, the caller closes the position
        close_side = "sell" if side == 1 else "buy"
        try:
            half = 0.0
            if partial_px:
                half = float(self.ex.amount_to_precision(symbol, qty / 2))
                if half < (self.ex.market(symbol)["limits"]["amount"]["min"] or 0):
                    half = 0.0
                else:
                    self.ex.create_order(symbol, "limit", close_side, half, self._px(symbol, partial_px),
                                         {"reduceOnly": True})
            rest = float(self.ex.amount_to_precision(symbol, qty - half))
            if tp and rest > 0:
                self.ex.create_order(symbol, "limit", close_side, rest, self._px(symbol, tp), {"reduceOnly": True})
        except Exception as e:  # the stop is already in place, so this isn't fatal
            log.warning("%s take-profit тавьж чадсангүй: %s", symbol, e)

    def _stops(self, symbol):
        out = []
        for o in self.ex.fetch_open_orders(symbol, params={"trigger": True}):
            info = o.get("info", {})
            kind = str(info.get("orderType") or info.get("type") or o.get("type") or "").upper()
            if o.get("stopLossPrice") or kind in ("STOP_MARKET", "STOP"):
                out.append(o)
        return out

    def has_stop(self, symbol):
        return bool(self._stops(symbol))

    def place_stop(self, symbol, side, qty, stop):
        self._stop_order(symbol, side, qty, stop)

    def move_stop(self, symbol, side, qty, new_stop):
        old = self._stops(symbol)
        new = self._stop_order(symbol, side, qty, new_stop)   # never be without a stop
        for o in old:
            if o["id"] != new.get("id"):
                try:
                    self.ex.cancel_order(o["id"], symbol, params={"trigger": True})
                except Exception as e:
                    log.warning("%s хуучин stop цуцалж чадсангүй: %s", symbol, e)

    def cancel_protection(self, symbol):
        for params in ({"trigger": True}, {}):
            try:
                self.ex.cancel_all_orders(symbol, params=params)
            except Exception as e:
                log.debug("cancel %s %s: %s", symbol, params, e)

    def close(self, symbol):
        p = self.positions().get(symbol)
        px = None
        if p:
            order = self.ex.create_order(symbol, "market", "sell" if p["side"] == 1 else "buy", p["qty"],
                                         None, {"reduceOnly": True})
            px = float(order.get("average") or self.price(symbol))
        self.cancel_protection(symbol)
        return px

    def closed_result(self, symbol, trade):
        fills = self.ex.fetch_my_trades(symbol, since=int(trade["opened_ts"]) - 60_000)
        close_side = "sell" if trade["side"] == "long" else "buy"
        pnl, fees, qty, notional = 0.0, 0.0, 0.0, 0.0
        for f in fills:
            pnl += float(f.get("info", {}).get("realizedPnl") or 0)
            if f.get("fee") and f["fee"].get("currency") in ("USDT", None):
                fees += float(f["fee"].get("cost") or 0)
            if f["side"] == close_side:
                qty += f["amount"]
                notional += f["amount"] * f["price"]
        exit_px = notional / qty if qty else None
        return exit_px, pnl - fees

    # ---- funding carry (spot long + perp short) ----
    def _spot_symbol(self, symbol):
        if not self.spot:
            raise RuntimeError("Энэ горимд spot данс алга тул carry ажиллахгүй")
        return f"{_base(symbol)}/USDT"

    def carry_open(self, symbol, notional):
        ssym = self._spot_symbol(symbol)
        px = self.price(symbol)
        qty, why = self.check_size(symbol, notional / px, px)
        if qty <= 0:
            raise ValueError(why)
        free_usdt = float(self.spot.fetch_balance()["free"].get("USDT", 0))
        if free_usdt < qty * px * 1.01:
            raise ValueError(f"spot данс дээр USDT хүрэлцэхгүй ({free_usdt:.2f})")
        so = self.spot.create_order(ssym, "market", "buy", qty)
        try:
            self._setup(symbol, leverage=1)
            po = self.ex.create_order(symbol, "market", "sell", qty)
        except Exception:
            log.exception("carry perp short амжилтгүй -> spot-ыг буцааж зарж байна")
            self.spot.create_order(ssym, "market", "sell", float(self.spot.amount_to_precision(ssym, qty * 0.998)))
            raise
        spot_px = float(so.get("average") or px)
        perp_px = float(po.get("average") or px)
        return {"qty": qty, "spot_px": spot_px, "perp_px": perp_px, "fees": qty * px * (SPOT_FEE + TAKER)}

    def carry_close(self, symbol, qty):
        ssym = self._spot_symbol(symbol)
        pos = self.positions().get(symbol)
        perp_px = spot_px = None
        if pos and pos["side"] == -1:
            po = self.ex.create_order(symbol, "market", "buy", pos["qty"], None, {"reduceOnly": True})
            perp_px = float(po.get("average") or self.price(symbol))
        free = float(self.spot.fetch_balance()["free"].get(_base(symbol), 0))
        sell = float(self.spot.amount_to_precision(ssym, min(free, qty)))
        if sell > 0:
            so = self.spot.create_order(ssym, "market", "sell", sell)
            spot_px = float(so.get("average") or self.price(symbol))
        px = self.price(symbol)
        return {"spot_px": spot_px or px, "perp_px": perp_px or px, "fees": qty * px * (SPOT_FEE + TAKER)}

    def carry_legs(self, symbol):
        pos = self.positions().get(symbol)
        spot_qty = float(self.spot.fetch_balance()["total"].get(_base(symbol), 0)) if self.spot else 0.0
        return {"spot_qty": spot_qty, "perp_qty": pos["qty"] if pos and pos["side"] == -1 else 0.0}

    def carry_funding(self, symbol, since_ms):
        rows = self.ex.fetch_funding_history(symbol, since=since_ms + 1)
        total = sum(float(r.get("amount") or 0) for r in rows)
        last = max([since_ms] + [r["timestamp"] for r in rows])
        return total, last


def make_broker(st: Storage, account):
    return PaperBroker(st, account) if mkt().mode == "signal" else BinanceBroker(st, account)
