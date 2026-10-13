"""The always-on loop for one market: ONE brain, MANY hands.

The brain (market-wide) reads candles and news, computes one signal per
symbol per candle, researches and self-improves. The hands are the people's
accounts (core/account.py): every signal is applied to each account with that
account's own money, risk limits, pending orders, kill switch and messages.
One person's broken key or loss never stops the others.

Every LOOP_SECONDS:  people from .env (every 5 min), Telegram commands, and per
                     account: equity / kill switch / drawdown, pending limit
                     entries, every position must have a stop.
Every closed candle: fresh data -> signal (once) -> per account: exits
                     (flip/time/breakeven/trail) and new limit entries.
Every hour:          calendar every 6h; market extras per account (crypto: carry).
Once a day:          research + self-improvement (walk-forward -> vault -> shadow).
Once a week:         a report to each person.
"""
import ctypes
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

import markets

from . import (events, exits, improver, news, research, review, reward, risk_guard, risk_ladder,
               telegram_bot)
from . import messages as M
from . import users as users_mod
from .account import Account
from .context import mkt
from .features import add_static_features
from .library import approved
from .notifier import notify
from .settings import IMPROVE_HOUR_UTC, LIVE_CONFIRM, LOOP_SECONDS, NEWS_EVERY_MIN, base_asset
from .storage import Storage, now_ms
from .strategy import compute, load_params, source_names, vault_score

log = logging.getLogger(__name__)
SIDE_NAME = {1: "long", -1: "short"}
SIDE_NUM = {"long": 1, "short": -1}
DAY_MS = 86_400_000
MAX_ENTRY_RETRIES = 5
USERS_EVERY_SEC = 300


def keep_awake(on: bool = True) -> None:
    """Ask Windows not to sleep while the bot runs (closing the lid can still sleep)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def _review_async(trade: dict, user):
    def work():
        lesson = review.review_trade(Storage(), trade["id"])
        if lesson:
            notify(M.review(trade["id"], trade["symbol"], SIDE_NUM[trade["side"]], lesson), to=user)
    threading.Thread(target=work, daemon=True).start()


class Trader:
    """Runs the active market (see core.context) for every person who trades it."""

    def __init__(self):
        self.m = mkt()
        self.st = Storage()
        self.check_mode()
        self.params = load_params()
        self.tf_ms = self.m.tf_ms()
        self.last_news = 0.0
        self.last_equity_log = {}
        self.last_users = 0.0
        self.accounts: dict[str, Account] = {}
        self.view = {}            # name -> (equity, day, positions) from this tick
        self.acc: Account | None = None
        self.broker = None
        self.refresh_accounts(announce=False)

    # ================= people =================
    def use(self, acc: Account):
        """Work on this person's account for the following steps."""
        self.acc, self.broker = acc, acc.broker

    def to(self):
        return self.acc.user

    def refresh_accounts(self, announce: bool = True):
        self.last_users = time.time()
        wanted = {u.name: u for u in self.m.traders(users_mod.load_users(markets.AVAILABLE))}
        for name in [n for n in self.accounts if n not in wanted]:
            self.accounts.pop(name)
            notify(f"👤 {name}: {self.m.label}-ээс хасагдлаа (.env-ээс устгагдсан эсвэл түлхүүр алга). "
                   "Нээлттэй position-ууд биржид stop/tp-тэйгээ үлдэнэ.")
        for name, u in wanted.items():
            if name in self.accounts:
                self.accounts[name].user = u   # pick up changed settings (risk, chat id)
                continue
            try:
                self.accounts[name] = Account(u, self.st)
            except Exception as e:
                notify(f"🚨 {name}: {self.m.label} данс холбогдсонгүй: {e}", key=f"accerr{name}", every_sec=3600)
                continue
            if announce:
                notify(f"👤 Шинэ хэрэглэгч: {name} ({self.m.label}, {self.m.mode})")
                notify(f"👋 Сайн байна уу, {name}! Таны {self.m.label} данс ботод холбогдлоо. /help", to=u)
        if not self.accounts:
            log.warning("%s: арилжаалах хэрэглэгч алга (.env USER<n>_MARKETS, түлхүүрүүдийг шалгана уу)", self.m.name)

    def primary(self) -> Account | None:
        admins = [a for a in self.accounts.values() if a.user.is_admin]
        return (admins or list(self.accounts.values()) or [None])[0]

    # ================= safety =================
    def check_mode(self):
        if self.m.mode in ("signal", "demo", "testnet"):
            if self.st.get("paper_started") is None:
                self.st.set("paper_started", now_ms())
            return
        if self.m.mode == "live":
            days = (now_ms() - (self.st.get("paper_started") or now_ms())) / DAY_MS
            ok, why = risk_guard.live_trading_allowed(days, LIVE_CONFIRM)
            if not ok:
                raise SystemExit("⛔ " + why)
            if vault_score() <= 0:
                raise SystemExit("⛔ Одоогийн стратеги хамгийн сүүлийн (vault) өгөгдөл дээр эерэг оноо аваагүй "
                                 "байна. Бодит мөнгө эрсдэлд оруулахгүй.")
            return
        raise SystemExit(f"Үл мэдэгдэх mode={self.m.mode} ({self.m.name})")

    def day_state(self, equity: float) -> dict:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        d = self.acc.get("day")
        if not d or d["date"] != today:
            if d:
                change = equity / d["start_equity"] - 1 if d["start_equity"] else 0
                dop = self.acc.get("dopamine", 0.0)
                day_start = int(datetime.strptime(d["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                n_closed = sum(1 for t in self.st.closed_trades(self.m.mode, 200, self.acc.name)
                               if (t["closed_ts"] or 0) >= day_start)
                notify(M.daily(d["date"], change, equity - d["start_equity"], equity, dop, reward.mood(dop), n_closed),
                       to=self.to())
            d = {"date": today, "start_equity": equity, "halted": False, "trades": 0}
            self.acc.set("day", d)
        return d

    def peak(self, equity: float) -> float:
        p = max(self.acc.get("peak_equity", 0.0), equity)
        self.acc.set("peak_equity", p)
        return p

    def paused(self) -> bool:
        return bool(self.acc.get("paused", False))

    # ================= bookkeeping =================
    def open_trades(self):
        return self.st.open_trades(self.m.mode, self.acc.name)

    def finish_trade(self, trade: dict, reason: str | None, exit_px=None, pnl=None):
        if exit_px is None or pnl is None:
            exit_px, pnl = self.broker.closed_result(trade["symbol"], trade)
        if exit_px is None:
            exit_px = self.broker.price(trade["symbol"])
            pnl = SIDE_NUM[trade["side"]] * trade["qty"] * (exit_px - trade["entry"])
        if reason is None:  # closed on the exchange: stop or take-profit?
            near_tp = trade["tp"] and abs(exit_px - trade["tp"]) < abs(exit_px - trade["stop"])
            reason = "tp" if near_tp else ("stop_after_partial" if trade.get("partial_done") else "stop")
        r_mult = pnl / trade["risk_usd"] if trade["risk_usd"] else 0.0
        rew = reward.trade_reward(r_mult, reason)
        dop = reward.update_dopamine(self.acc.get("dopamine", 0.0), rew)
        self.acc.set("dopamine", dop)
        self.st.update_trade(trade["id"], status="closed", closed_ts=now_ms(), exit=exit_px, pnl=pnl,
                             r_multiple=r_mult, reward=rew, exit_reason=reason)
        self.broker.cancel_protection(trade["symbol"])  # leftover stop/TP must not hit a future position
        hours = (now_ms() - trade["opened_ts"]) / 3_600_000
        notify(M.closed(trade["symbol"], SIDE_NUM[trade["side"]], trade["entry"], exit_px, pnl, r_mult, reason,
                        hours, rew, dop, reward.mood(dop)), to=self.to())
        _review_async(trade, self.to())

    def directional(self, positions: dict) -> dict:
        return {s: p for s, p in positions.items() if s in self.m.symbols}

    def reconcile(self, positions: dict):
        for t in self.open_trades():
            sym = t["symbol"]
            if sym not in positions:
                self.finish_trade(t, None)
                continue
            side = SIDE_NUM[t["side"]]
            if not t["partial_done"] and positions[sym]["qty"] < t["qty"] * 0.75:
                self.st.update_trade(t["id"], partial_done=1)
                notify(M.partial_taken(sym, side), to=self.to())
            if not self.broker.has_stop(sym):
                # Rule: every position must have a stop on the exchange.
                try:
                    self.broker.place_stop(sym, side, positions[sym]["qty"], t["stop"])
                    notify(f"⚠️ {sym}: stop-loss алга байсныг дахин тавилаа", key=f"stop{sym}", every_sec=600,
                           to=self.to())
                except Exception as e:
                    notify(f"🚨 {sym}: stop тавьж чадсангүй ({e}) -> хааж байна", to=self.to())
                    self.broker.close(sym)
                    self.finish_trade(t, "no_stop")
        known = {t["symbol"] for t in self.open_trades()} | \
                {c["symbol"] for c in self.st.open_carries(self.m.mode, self.acc.name)}
        for sym in positions:
            if sym not in known and not self.acc.get(f"pending:{sym}"):
                notify(f"⚠️ {sym}: ботын мэдэхгүй position байна (гараар нээсэн?). Бот үүнд хүрэхгүй.",
                       key=f"unknown{sym}", every_sec=3600, to=self.to())

    def close_everything(self, reason: str, include_carry: bool):
        for sym in self.m.symbols:
            pend = self.acc.get(f"pending:{sym}")
            if pend:
                self.broker.cancel_entry(sym, pend)
                self.acc.set(f"pending:{sym}", None)
        for t in self.open_trades():
            self.broker.close(t["symbol"])
            self.finish_trade(t, reason)
        if include_carry:
            for c in self.st.open_carries(self.m.mode, self.acc.name):
                self.m.close_carry(self, c, reason)

    def kill_switch(self, equity: float, day: dict):
        if not day["halted"] and risk_guard.daily_loss_hit(day["start_equity"], equity):
            day["halted"] = True
            self.acc.set("day", day)
            self.close_everything("kill_switch", include_carry=True)
            notify(M.kill_switch(risk_guard.MAX_DAILY_LOSS), to=self.to())

    # ================= entries =================
    def open_exposure(self, positions: dict):
        """(count, notional, risk) of directional positions + pending entries."""
        count, notional, risk = 0, 0.0, 0.0
        for t in self.open_trades():
            if t["symbol"] in positions:
                p = positions[t["symbol"]]
                count += 1
                notional += p["qty"] * p["entry"]
                risk += max(0.0, SIDE_NUM[t["side"]] * (t["entry"] - t["stop"])) * p["qty"]
        for sym in self.m.symbols:
            pend = self.acc.get(f"pending:{sym}")
            if pend:
                count += 1
                notional += pend["qty"] * pend["limit"]
                risk += pend["qty"] * pend["dist"]
        return count, notional, risk

    def correlation(self, sym: str, side: int) -> float:
        """Highest return correlation with an open/pending position on the same side."""
        others = [t["symbol"] for t in self.open_trades() if SIDE_NUM[t["side"]] == side and t["symbol"] != sym]
        others += [s for s in self.m.symbols if s != sym and (self.acc.get(f"pending:{s}") or {}).get("side") == side]
        if not others:
            return 0.0
        rets = {}
        for s in [sym, *others]:
            df = self.m.load_frame(self.st, s)
            rets[s] = df["close"].iloc[-720:].pct_change()
        return max(float(np.nan_to_num(rets[sym].corr(rets[o]))) for o in others)

    def risk_per_trade(self) -> float | None:
        """The person's own choice, capped by the market's risk ladder (if it has one)."""
        mine = self.acc.user.risk_per_trade
        lad = risk_ladder.config()
        if not lad:
            return mine
        rs = [t["r_multiple"] for t in self.st.closed_trades(self.m.mode, 1000, self.acc.name)
              if t.get("r_multiple") is not None]
        state, msg = risk_ladder.step(self.acc.get("ladder") or {}, rs, lad)
        self.acc.set("ladder", state)
        if msg:
            notify(msg, to=self.to())
        step = risk_ladder.current(state, lad)
        return min(mine, step) if mine else step

    def try_enter(self, sym: str, sig: dict, positions: dict, equity: float, day: dict) -> str | None:
        """Places a limit entry. Returns the risk-guard reason if the signal was refused."""
        side = sig["signal"]
        dist = self.params["stop_atr"] * sig["atr"]
        limit = sig["close"] - side * self.params["entry_offset_atr"] * sig["atr"]
        count, notional, risk = self.open_exposure(positions)
        ok, why = self.m.spread_ok(sym)
        if not ok:
            return why
        dec = risk_guard.check_entry(
            equity=equity, entry=limit, stop=limit - side * dist, side=side, open_positions=count,
            open_notional=notional, trades_today=day["trades"], halted=day["halted"],
            peak_equity=self.acc.get("peak_equity", equity), open_risk=risk,
            max_corr_same_side=self.correlation(sym, side),
            blackout=events.blackout(self.st, symbol=sym), paused=self.paused(),
            risk_per_trade=self.risk_per_trade())
        if not dec.allowed:
            return dec.reason
        try:
            pend = self.broker.place_entry(sym, side, dec.qty, limit)
        except Exception as e:
            notify(f"❌ {sym} захиалга тавьж чадсангүй: {e}", key=f"openerr{sym}", every_sec=1800, to=self.to())
            return None
        pend.update(side=side, dist=dist, bar_ts=sig["bar_ts"], sig=sig, retries=0,
                    risk_fraction=dec.risk_fraction, expires=sig["bar_ts"] + 2 * self.tf_ms,
                    limit=pend.get("limit", limit), qty=pend.get("qty", dec.qty))
        day["trades"] += 1
        self.acc.set("day", day)
        if pend["status"] == "filled":
            self.on_filled(sym, pend, pend["filled"], pend["avg"])
        else:
            self.acc.set(f"pending:{sym}", pend)
            notify(M.limit_placed(sym, side, pend["limit"], sig["score"], self.params["entry_threshold"]), to=self.to())
        return None

    def on_filled(self, sym: str, pend: dict, qty: float, avg: float):
        side, dist = pend["side"], pend["dist"]
        stop = avg - side * dist
        tp = avg + side * dist * self.params["tp_r"]
        partial = exits.partial_price(avg, side, dist, self.params)
        self.acc.set(f"pending:{sym}", None)
        try:
            self.broker.protect(sym, side, qty, stop, tp, partial)
        except Exception as e:
            notify(f"🚨 {sym}: stop тавьж чадсангүй ({e}) -> шууд хааж байна", to=self.to())
            self.broker.close(sym)
            return
        sig = pend["sig"]
        self.st.insert_trade({
            "mode": self.m.mode, "user": self.acc.name, "symbol": sym, "side": SIDE_NAME[side], "qty": qty,
            "entry": avg, "stop": stop, "tp": tp, "risk_usd": qty * dist, "opened_ts": now_ms(),
            "opened_bar_ts": pend["bar_ts"], "status": "open", "signal_json": json.dumps(sig), "dist": dist,
            "best": avg, "partial_done": 0, "regime": sig.get("regime"),
        })
        weights = {k: self.params.get("w_range_mr" if k == "mean_rev" else f"w_{k}", 0.0) for k in sig["parts"]}
        notify(M.opened(sym, side, avg, qty, stop, tp, partial, pend.get("risk_fraction", 0.0), sig["parts"], weights),
               to=self.to())

    def process_pending(self):
        for sym in self.m.symbols:
            pend = self.acc.get(f"pending:{sym}")
            if not pend:
                continue
            s = self.broker.entry_status(sym, pend)
            if s["status"] == "filled":
                self.on_filled(sym, pend, s["filled"], s["avg"])
            elif now_ms() > pend["expires"]:
                s = self.broker.cancel_entry(sym, pend)
                if s["filled"] > 0:
                    self.on_filled(sym, pend, s["filled"], s["avg"])
                else:
                    self.acc.set(f"pending:{sym}", None)
                    log.info("%s/%s limit захиалга биелээгүй, цуцаллаа", self.acc.name, sym)
            elif s["status"] == "canceled":
                if s["filled"] > 0:
                    self.on_filled(sym, pend, s["filled"], s["avg"])
                elif pend["retries"] < MAX_ENTRY_RETRIES:  # post-only rejected because price moved; rejoin the book
                    try:
                        new = self.broker.place_entry(sym, pend["side"], pend["qty"], pend["limit"])
                        pend.update(id=new.get("id"), retries=pend["retries"] + 1, status=new["status"])
                        if new["status"] == "filled":
                            self.on_filled(sym, pend, new["filled"], new["avg"])
                            continue
                        self.acc.set(f"pending:{sym}", pend)
                    except Exception as e:
                        log.warning("%s/%s дахин захиалга: %s", self.acc.name, sym, e)
                        self.acc.set(f"pending:{sym}", None)
                else:
                    self.acc.set(f"pending:{sym}", None)

    # ================= candles (brain) =================
    def signal_for(self, symbol: str):
        self.m.sync_symbol(self.st, symbol)
        df = self.m.load_frame(self.st, symbol)
        if len(df) < 300:
            return None
        df = df.iloc[-3000:].copy()
        if self.params["w_news"]:
            df.loc[df.index[-1], "news"] = news.current_sentiment(self.st, base_asset(symbol))
        df = add_static_features(df, approved())
        f = compute(df, self.params)
        last, bar = f.iloc[-1], df.iloc[-1]
        parts = {k: round(float(last[k]), 3) for k in source_names()}
        return {"bar_ts": int(df.index[-1]), "signal": int(last["signal"]), "score": round(float(last["score"]), 3),
                "atr": float(last["atr"]), "regime": int(last["regime"]), "volatile": bool(last["volatile"]),
                "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "parts": parts}

    def manage_open(self, t: dict, sig: dict, positions: dict) -> bool:
        """Exit / tighten-stop logic for an open trade. Returns True if closed."""
        sym, side = t["symbol"], SIDE_NUM[t["side"]]
        held = (sig["bar_ts"] - t["opened_bar_ts"]) // self.tf_ms
        reason = "flip" if sig["signal"] == -side else "time" if held >= self.params["max_hold_bars"] else None
        if reason:
            self.broker.close(sym)
            self.finish_trade(t, reason)
            return True
        pos = {"side": side, "entry": t["entry"], "stop": t["stop"], "dist": t["dist"] or abs(t["entry"] - t["stop"]),
               "best": t["best"] or t["entry"]}
        new_stop = exits.update_stop(pos, sig["high"], sig["low"], sig["close"], sig["atr"], self.params)
        self.st.update_trade(t["id"], best=pos["best"])
        if (new_stop - t["stop"]) * side > 0.02 * pos["dist"]:
            try:
                self.broker.move_stop(sym, side, positions[sym]["qty"], new_stop)
                self.st.update_trade(t["id"], stop=new_stop)
                notify(M.stop_moved(sym, side, t["stop"], new_stop, t["entry"], positions[sym]["qty"]),
                       key=f"move{sym}", every_sec=3 * 3600, to=self.to())
            except Exception as e:
                log.warning("%s/%s stop шилжүүлж чадсангүй (%s) -> үнэ аль хэдийн давсан тул хааж байна",
                            self.acc.name, sym, e)
                self.broker.close(sym)
                self.finish_trade(t, "trail")
                return True
        return False

    def new_signals(self, bar_now: int) -> tuple[dict, bool]:
        """Brain: one signal per symbol for each newly closed candle (computed once for everyone).
        Returns ({symbol: signal}, done) where done = every symbol has reached bar_now."""
        sigs, done = {}, True
        for sym in self.m.symbols:
            sig = self.signal_for(sym)
            if not sig or (sig["bar_ts"] < bar_now and self.m.is_open(bar_now)):
                done = False  # exchange hasn't published the candle yet; retry next tick (none come when closed)
            if not sig or sig["bar_ts"] == self.st.get(f"bar:{sym}"):
                continue
            self.st.set(f"bar:{sym}", sig["bar_ts"])
            regime = {1: "trend", -1: "range", 0: "дунд"}[sig["regime"]]
            log.info("%s оноо=%+.2f дохио=%s regime=%s%s %s", sym, sig["score"], sig["signal"], regime,
                     " (хэт хэлбэлзэлтэй)" if sig["volatile"] else "", sig["parts"])
            sigs[sym] = sig
        return sigs, done

    def apply_signals(self, sigs: dict, positions: dict, equity: float, day: dict):
        """Hands: act on the brain's signals for the current account."""
        open_trades = {t["symbol"]: t for t in self.open_trades()}
        refused = {}
        for sym, sig in sigs.items():
            t = open_trades.get(sym)
            if t and sym in positions:
                if self.manage_open(t, sig, positions):
                    positions.pop(sym, None)
                continue
            if not self.m.is_open():  # e.g. FX at the weekend: manage open trades, open nothing new
                continue
            if sig["signal"] != 0 and sym not in positions and not self.acc.get(f"pending:{sym}"):
                why = self.try_enter(sym, sig, positions, equity, day)
                if why:
                    refused.setdefault(why, []).append((sym, sig["signal"], sig["score"]))
        for why, items in refused.items():  # one message per reason; the same list isn't repeated for 3h
            key = "skip:" + why + ":" + ",".join(s for s, _, _ in items)
            notify(M.skipped(items, why), key=key, every_sec=3 * 3600, to=self.to())

    # ================= Telegram =================
    def status_text(self, equity: float) -> str:
        day = self.acc.get("day") or {}
        dop = self.acc.get("dopamine", 0.0)
        start = day.get("start_equity") or equity
        trades = self.open_trades()
        state = "⏸ Түр зогссон (/resume)" if self.paused() else ("🛑 Kill switch" if day.get("halted") else "▶️ Ажиллаж байна")
        lines = [
            f"📊 СТАТУС · {datetime.now().strftime('%m/%d %H:%M')}",
            "",
            f"💰 Данс:     {equity:,.2f} USDT",
            f"📅 Өнөөдөр:  {equity / start - 1:+.2%} ({equity - start:+,.2f} $)",
            f"🧠 Допамин:  {dop:+.2f} {reward.mood(dop)}",
            f"{state} · position {len(trades)}/{risk_guard.limits()['max_open_positions']}",
        ]
        if trades:
            lines += ["", "━━━ НЭЭЛТТЭЙ POSITION ━━━"]
        for t in trades:
            side = SIDE_NUM[t["side"]]
            try:
                px = self.broker.price(t["symbol"])
            except Exception:
                px = t["entry"]
            pnl = side * t["qty"] * (px - t["entry"])
            r = pnl / t["risk_usd"] if t["risk_usd"] else 0.0
            hours = (now_ms() - t["opened_ts"]) / 3_600_000
            moved = (px / t["entry"] - 1) * 100
            lines += [
                "",
                f"{'🟢' if pnl >= 0 else '🔴'} {base_asset(t['symbol'])} {t['side'].upper()} · {hours:.0f} цаг"
                + (" · 💰 хагас авсан" if t["partial_done"] else ""),
                f"   Орсон:  {M.px(t['entry'])}",
                f"   Одоо:   {M.px(px)} ({moved:+.2f}%)",
                f"   Ашиг:   {pnl:+,.2f} $ ({r:+.2f}R)",
                f"   ❌ Stop: {M.px(t['stop'])} ({abs(t['stop'] / px - 1) * 100:.1f}% зайтай)",
                f"   🎯 TP:   {M.px(t['tp'])} ({abs(t['tp'] / px - 1) * 100:.1f}% зайтай)",
            ]
        pend = [(s, self.acc.get(f"pending:{s}")) for s in self.m.symbols]
        pend = [(s, p) for s, p in pend if p]
        if pend:
            lines += ["", "━━━ ХҮЛЭЭГДЭЖ БУЙ ЗАХИАЛГА ━━━"]
            lines += [f"🎯 {base_asset(s)} {SIDE_NAME[p['side']].upper()} limit @ {M.px(p['limit'])}" for s, p in pend]
        carries = self.st.open_carries(self.m.mode, self.acc.name)
        if carries:
            lines += ["", "━━━ FUNDING CARRY ━━━"]
            lines += [f"🏦 {base_asset(c['symbol'])}: funding {c['funding'] or 0:+.2f} $" for c in carries]
        if not trades and not pend:
            lines += ["", "Нээлттэй position алга. Дохио хүлээж байна."]
        return "\n".join(lines)

    def handle_commands(self):
        for user, cmd in telegram_bot.poll(self.st):
            if user.is_admin and cmd in ("/users", "/emergency", "/emergency_yes"):
                self.admin_command(user, cmd)
                continue
            acc = self.accounts.get(user.name)
            if acc is None:
                notify(f"Таны данс {self.m.label}-д идэвхгүй байна.", to=user)
                continue
            self.use(acc)
            equity = self.view.get(user.name, (None,))[0] or self.broker.equity()
            if cmd == "/status":
                notify(self.status_text(equity), to=user)
            elif cmd == "/stop":
                acc.set("paused", True)
                notify("⏸ Шинэ арилжаа зогслоо. Нээлттэй position stop/tp-ээрээ үргэлжилнэ. /resume гэж эхлүүлнэ.", to=user)
            elif cmd == "/resume":
                acc.set("paused", False)
                notify("▶️ Дахин эхэллээ.", to=user)
            elif cmd == "/closeall":
                acc.set("closeall_asked", now_ms())
                notify("Бүх position (carry орно) хаах уу? 2 минутын дотор /closeall_yes гэж бичнэ үү.", to=user)
            elif cmd == "/closeall_yes":
                if now_ms() - acc.get("closeall_asked", 0) < 120_000:
                    self.close_everything("manual", include_carry=True)
                    acc.set("paused", True)
                    notify("🧹 Бүгдийг хаалаа, таны бот зогссон төлөвт байна (/resume).", to=user)
                else:
                    notify("Хугацаа дууссан. Эхлээд /closeall гэж бичнэ үү.", to=user)
            elif cmd == "/report":
                notify(review.weekly_report(self.st, self.m.mode, user.name), to=user)
            elif cmd == "/resetpeak":
                acc.set("peak_equity", equity)
                notify(f"Оргил equity-г {equity:.2f} болгож шинэчиллээ. Drawdown хязгаар дахин тоологдоно.", to=user)
            else:
                notify(telegram_bot.HELP + (telegram_bot.ADMIN_HELP if user.is_admin else ""), to=user)

    def admin_command(self, admin, cmd: str):
        if cmd == "/users":
            lines = [f"👥 {self.m.label} · {self.m.mode}"]
            for name, acc in self.accounts.items():
                eq = self.view.get(name, (None,))[0]
                n_open = len(self.st.open_trades(self.m.mode, name))
                paused = " ⏸" if acc.get("paused") else ""
                lines.append(f"• {name}{' 👑' if acc.user.is_admin else ''}: "
                             f"{eq:,.2f} $ · {n_open} position{paused}" if eq is not None else f"• {name}: —")
            notify("\n".join(lines), to=admin)
        elif cmd == "/emergency":
            self.st.set("emergency_asked", now_ms())
            notify("🚨 БҮХ хүний бүх position-ыг хааж, ботыг зогсоох уу? 2 минутын дотор /emergency_yes.", to=admin)
        elif cmd == "/emergency_yes":
            if now_ms() - self.st.get("emergency_asked", 0) > 120_000:
                notify("Хугацаа дууссан. Эхлээд /emergency гэж бичнэ үү.", to=admin)
                return
            for acc in self.accounts.values():
                self.use(acc)
                try:
                    self.close_everything("manual", include_carry=True)
                except Exception as e:
                    notify(f"🚨 {acc.name}: хааж чадсангүй: {e}", to=admin)
                acc.set("paused", True)
                notify("🚨 Админ яаралтай зогсоолт хийлээ: бүх position хаагдаж, бот зогссон.", to=acc.user)
            notify("🚨 Бүх хүний арилжаа зогсоож, хаалаа. Хүн бүр /resume-ээр дахин эхлүүлнэ.", to=admin)

    # ================= periodic jobs =================
    def maybe_news(self):
        if time.time() - self.last_news < NEWS_EVERY_MIN * 60:
            return
        self.last_news = time.time()
        try:
            n = news.fetch_headlines(self.st)
            k = news.score_news(self.st)
            log.info("Мэдээ: %d гарчиг, %d үнэлгээ", n, k)
        except Exception as e:
            log.warning("Мэдээ уншихад алдаа: %s", e)

    def maybe_hourly(self):
        hour = now_ms() // 3_600_000
        if self.st.get("hourly_done") == hour:
            return
        self.st.set("hourly_done", hour)
        if hour % 6 == 0 or not self.st.get("calendar"):
            try:
                events.refresh(self.st)
            except Exception as e:
                log.warning("Календарь: %s", e)
        for name, acc in self.accounts.items():   # market-specific hourly jobs per person (crypto: carry)
            if name in self.view:
                self.use(acc)
                equity, day, _ = self.view[name]
                self.m.hourly(self, equity, day)

    def maybe_daily(self):
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if now.hour < IMPROVE_HOUR_UTC or self.st.get("improved_on") == today:
            return
        self.st.set("improved_on", today)
        try:
            self.m.daily_sync(self.st)
        except Exception as e:
            log.warning("Өдрийн өгөгдөл: %s", e)
        try:
            new = research.research(self.st)
            if new:
                notify(f"📚 Интернэтээс {len(new)} шинэ стратеги олдлоо, backtest хийж байна")
        except Exception as e:
            log.warning("Судалгаа: %s", e)
        lead = self.primary()   # the improver follows the main admin's dopamine
        rep = improver.improve(self.st, self.m.symbols, dopamine=lead.get("dopamine", 0.0) if lead else 0.0)
        for s in rep.get("new_shadows", []):
            notify(f"🧪 Шинэ shadow {s['id']}: vault оноо {s['vault']['score']}, бодит цаг хугацаанд туршиж эхэллээ")
        if rep.get("shadow_promoted"):
            notify(f"🧠 Shadow {rep['shadow_promoted']} ирээдүйн өгөгдөл дээр давж гарсан тул идэвхтэй тохиргоо боллоо")
        self.params = load_params()
        if now.weekday() == 0 and self.st.get("weekly_on") != today:
            self.st.set("weekly_on", today)
            for acc in self.accounts.values():
                notify(review.weekly_report(self.st, self.m.mode, acc.name), to=acc.user)

    # ================= main loop =================
    def account_tick(self, acc: Account):
        """Housekeeping for one person's account."""
        self.use(acc)
        equity = self.broker.equity()
        day = self.day_state(equity)
        self.peak(equity)
        if time.time() - self.last_equity_log.get(acc.name, 0) > 300:
            self.last_equity_log[acc.name] = time.time()
            self.st.log_equity(self.m.mode, equity, acc.get("dopamine", 0.0), acc.name)
        self.process_pending()
        self.reconcile(self.directional(self.broker.positions()))
        self.kill_switch(equity, day)
        self.view[acc.name] = (equity, day, self.directional(self.broker.positions()))

    def for_each_account(self, fn):
        for acc in list(self.accounts.values()):
            try:
                fn(acc)
            except Exception as e:  # one person's problem never stops the others
                log.exception("%s: алдаа", acc.name)
                notify(f"🚨 {acc.name}: {e}", key=f"err{acc.name}", every_sec=1800, to=acc.user)
                notify(f"🚨 {acc.name}-ийн дансанд алдаа: {e}", key=f"aerr{acc.name}", every_sec=1800)

    def tick(self):
        if time.time() - self.last_users > USERS_EVERY_SEC:
            self.refresh_accounts()
        self.handle_commands()
        self.for_each_account(self.account_tick)
        self.maybe_news()
        self.maybe_hourly()
        bar_now = (now_ms() // self.tf_ms) * self.tf_ms - self.tf_ms  # last fully closed bar
        if self.st.get("checked_bar") != bar_now:
            sigs, done = self.new_signals(bar_now)
            if sigs:
                def act(acc):
                    if acc.name not in self.view:
                        return
                    self.use(acc)
                    equity, day, positions = self.view[acc.name]
                    self.apply_signals(sigs, positions, equity, day)
                self.for_each_account(act)
            if done:
                self.st.set("checked_bar", bar_now)
        self.maybe_daily()
        self.st.set("heartbeat", now_ms())  # the web admin page shows when each market last ran

    def any_pending(self) -> bool:
        return any(acc.get(f"pending:{s}") for acc in self.accounts.values() for s in self.m.symbols)

    def run(self):
        keep_awake(True)
        for acc in self.accounts.values():
            try:
                notify(M.startup(self.m.mode, self.m.symbols, acc.broker.equity()), to=acc.user)
            except Exception as e:
                log.warning("%s: %s", acc.name, e)
        errors = 0
        try:
            while True:
                try:
                    self.tick()
                    errors = 0
                except Exception as e:
                    errors += 1
                    log.exception("tick алдаа")
                    if errors >= 5:
                        notify(f"🚨 Дараалсан {errors} алдаа: {e}", key="errors", every_sec=1800)
                # check pending limit orders more often, so a fill gets its stop quickly
                time.sleep(5 if self.any_pending() else LOOP_SECONDS)
        finally:
            keep_awake(False)
