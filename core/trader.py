"""The always-on loop.

Every LOOP_SECONDS:  Telegram commands, equity / kill switch / drawdown,
                     pending limit entries, every position must have a stop.
Every closed candle: fresh data -> signal -> exits (flip/time/breakeven/trail)
                     -> maybe place a new limit entry.
Every hour:          funding carry; calendar every 6h; news every NEWS_EVERY_MIN.
Once a day:          research + self-improvement (walk-forward -> vault -> shadow).
Once a week:         report to Telegram.
"""
import ctypes
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

from . import (events, exits, improver, news, research, review, reward, risk_guard,
               telegram_bot)
from . import messages as M
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


def keep_awake(on: bool = True) -> None:
    """Ask Windows not to sleep while the bot runs (closing the lid can still sleep)."""
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def _review_async(trade: dict):
    def work():
        lesson = review.review_trade(Storage(), trade["id"])
        if lesson:
            notify(M.review(trade["id"], trade["symbol"], SIDE_NUM[trade["side"]], lesson))
    threading.Thread(target=work, daemon=True).start()


class Trader:
    """Runs one market (the active one, see core.context)."""

    def __init__(self):
        self.m = mkt()
        self.st = Storage()
        self.check_mode()
        self.broker = self.m.make_broker(self.st)
        self.params = load_params()
        self.tf_ms = self.m.tf_ms()
        self.last_news = 0.0
        self.last_equity_log = 0.0

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
        d = self.st.get("day")
        if not d or d["date"] != today:
            if d:
                change = equity / d["start_equity"] - 1 if d["start_equity"] else 0
                dop = self.st.get("dopamine", 0.0)
                day_start = int(datetime.strptime(d["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
                n_closed = sum(1 for t in self.st.closed_trades(self.m.mode, 200) if (t["closed_ts"] or 0) >= day_start)
                notify(M.daily(d["date"], change, equity - d["start_equity"], equity, dop, reward.mood(dop), n_closed))
            d = {"date": today, "start_equity": equity, "halted": False, "trades": 0}
            self.st.set("day", d)
        return d

    def peak(self, equity: float) -> float:
        p = max(self.st.get("peak_equity", 0.0), equity)
        self.st.set("peak_equity", p)
        return p

    def paused(self) -> bool:
        return bool(self.st.get("paused", False))

    # ================= bookkeeping =================
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
        dop = reward.update_dopamine(self.st.get("dopamine", 0.0), rew)
        self.st.set("dopamine", dop)
        self.st.update_trade(trade["id"], status="closed", closed_ts=now_ms(), exit=exit_px, pnl=pnl,
                             r_multiple=r_mult, reward=rew, exit_reason=reason)
        self.broker.cancel_protection(trade["symbol"])  # leftover stop/TP must not hit a future position
        hours = (now_ms() - trade["opened_ts"]) / 3_600_000
        notify(M.closed(trade["symbol"], SIDE_NUM[trade["side"]], trade["entry"], exit_px, pnl, r_mult, reason,
                        hours, rew, dop, reward.mood(dop)))
        _review_async(trade)

    def directional(self, positions: dict) -> dict:
        return {s: p for s, p in positions.items() if s in self.m.symbols}

    def reconcile(self, positions: dict):
        for t in self.st.open_trades(self.m.mode):
            sym = t["symbol"]
            if sym not in positions:
                self.finish_trade(t, None)
                continue
            side = SIDE_NUM[t["side"]]
            if not t["partial_done"] and positions[sym]["qty"] < t["qty"] * 0.75:
                self.st.update_trade(t["id"], partial_done=1)
                notify(M.partial_taken(sym, side))
            if not self.broker.has_stop(sym):
                # Rule: every position must have a stop on the exchange.
                try:
                    self.broker.place_stop(sym, side, positions[sym]["qty"], t["stop"])
                    notify(f"⚠️ {sym}: stop-loss алга байсныг дахин тавилаа", key=f"stop{sym}", every_sec=600)
                except Exception as e:
                    notify(f"🚨 {sym}: stop тавьж чадсангүй ({e}) -> хааж байна")
                    self.broker.close(sym)
                    self.finish_trade(t, "no_stop")
        known = {t["symbol"] for t in self.st.open_trades(self.m.mode)} | {c["symbol"] for c in self.st.open_carries(self.m.mode)}
        for sym in positions:
            if sym not in known and not self.st.get(f"pending:{sym}"):
                notify(f"⚠️ {sym}: ботын мэдэхгүй position байна (гараар нээсэн?). Бот үүнд хүрэхгүй.",
                       key=f"unknown{sym}", every_sec=3600)

    def close_everything(self, reason: str, include_carry: bool):
        for sym in self.m.symbols:
            pend = self.st.get(f"pending:{sym}")
            if pend:
                self.broker.cancel_entry(sym, pend)
                self.st.set(f"pending:{sym}", None)
        for t in self.st.open_trades(self.m.mode):
            self.broker.close(t["symbol"])
            self.finish_trade(t, reason)
        if include_carry:
            for c in self.st.open_carries(self.m.mode):
                self.m.close_carry(self, c, reason)

    def kill_switch(self, equity: float, day: dict):
        if not day["halted"] and risk_guard.daily_loss_hit(day["start_equity"], equity):
            day["halted"] = True
            self.st.set("day", day)
            self.close_everything("kill_switch", include_carry=True)
            notify(M.kill_switch(risk_guard.MAX_DAILY_LOSS))

    # ================= entries =================
    def open_exposure(self, positions: dict):
        """(count, notional, risk) of directional positions + pending entries."""
        count, notional, risk = 0, 0.0, 0.0
        for t in self.st.open_trades(self.m.mode):
            if t["symbol"] in positions:
                p = positions[t["symbol"]]
                count += 1
                notional += p["qty"] * p["entry"]
                risk += max(0.0, SIDE_NUM[t["side"]] * (t["entry"] - t["stop"])) * p["qty"]
        for sym in self.m.symbols:
            pend = self.st.get(f"pending:{sym}")
            if pend:
                count += 1
                notional += pend["qty"] * pend["limit"]
                risk += pend["qty"] * pend["dist"]
        return count, notional, risk

    def correlation(self, sym: str, side: int, positions: dict) -> float:
        """Highest return correlation with an open/pending position on the same side."""
        others = [t["symbol"] for t in self.st.open_trades(self.m.mode) if SIDE_NUM[t["side"]] == side and t["symbol"] != sym]
        others += [s for s in self.m.symbols if s != sym and (self.st.get(f"pending:{s}") or {}).get("side") == side]
        if not others:
            return 0.0
        rets = {}
        for s in [sym, *others]:
            df = self.m.load_frame(self.st, s)
            rets[s] = df["close"].iloc[-720:].pct_change()
        return max(float(np.nan_to_num(rets[sym].corr(rets[o]))) for o in others)

    def try_enter(self, sym: str, sig: dict, positions: dict, equity: float, day: dict) -> str | None:
        """Places a limit entry. Returns the risk-guard reason if the signal was refused."""
        side = sig["signal"]
        dist = self.params["stop_atr"] * sig["atr"]
        limit = sig["close"] - side * self.params["entry_offset_atr"] * sig["atr"]
        count, notional, risk = self.open_exposure(positions)
        dec = risk_guard.check_entry(
            equity=equity, entry=limit, stop=limit - side * dist, side=side, open_positions=count,
            open_notional=notional, trades_today=day["trades"], halted=day["halted"],
            peak_equity=self.st.get("peak_equity", equity), open_risk=risk,
            max_corr_same_side=self.correlation(sym, side, positions),
            blackout=events.blackout(self.st), paused=self.paused())
        if not dec.allowed:
            return dec.reason
        try:
            pend = self.broker.place_entry(sym, side, dec.qty, limit)
        except Exception as e:
            notify(f"❌ {sym} захиалга тавьж чадсангүй: {e}", key=f"openerr{sym}", every_sec=1800)
            return None
        pend.update(side=side, dist=dist, bar_ts=sig["bar_ts"], sig=sig, retries=0,
                    risk_fraction=dec.risk_fraction, expires=sig["bar_ts"] + 2 * self.tf_ms,
                    limit=pend.get("limit", limit), qty=pend.get("qty", dec.qty))
        day["trades"] += 1
        self.st.set("day", day)
        if pend["status"] == "filled":
            self.on_filled(sym, pend, pend["filled"], pend["avg"])
        else:
            self.st.set(f"pending:{sym}", pend)
            notify(M.limit_placed(sym, side, pend["limit"], sig["score"], self.params["entry_threshold"]))
        return None

    def on_filled(self, sym: str, pend: dict, qty: float, avg: float):
        side, dist = pend["side"], pend["dist"]
        stop = avg - side * dist
        tp = avg + side * dist * self.params["tp_r"]
        partial = exits.partial_price(avg, side, dist, self.params)
        self.st.set(f"pending:{sym}", None)
        try:
            self.broker.protect(sym, side, qty, stop, tp, partial)
        except Exception as e:
            notify(f"🚨 {sym}: stop тавьж чадсангүй ({e}) -> шууд хааж байна")
            self.broker.close(sym)
            return
        sig = pend["sig"]
        self.st.insert_trade({
            "mode": self.m.mode, "symbol": sym, "side": SIDE_NAME[side], "qty": qty, "entry": avg, "stop": stop, "tp": tp,
            "risk_usd": qty * dist, "opened_ts": now_ms(), "opened_bar_ts": pend["bar_ts"], "status": "open",
            "signal_json": json.dumps(sig), "dist": dist, "best": avg, "partial_done": 0, "regime": sig.get("regime"),
        })
        weights = {k: self.params.get("w_range_mr" if k == "mean_rev" else f"w_{k}", 0.0) for k in sig["parts"]}
        notify(M.opened(sym, side, avg, qty, stop, tp, partial, pend.get("risk_fraction", 0.0), sig["parts"], weights))

    def process_pending(self):
        for sym in self.m.symbols:
            pend = self.st.get(f"pending:{sym}")
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
                    self.st.set(f"pending:{sym}", None)
                    log.info("%s limit захиалга биелээгүй, цуцаллаа", sym)
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
                        self.st.set(f"pending:{sym}", pend)
                    except Exception as e:
                        log.warning("%s дахин захиалга: %s", sym, e)
                        self.st.set(f"pending:{sym}", None)
                else:
                    self.st.set(f"pending:{sym}", None)

    # ================= candles =================
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
                       key=f"move{sym}", every_sec=3 * 3600)
            except Exception as e:
                log.warning("%s stop шилжүүлж чадсангүй (%s) -> үнэ аль хэдийн давсан тул хааж байна", sym, e)
                self.broker.close(sym)
                self.finish_trade(t, "trail")
                return True
        return False

    def on_new_candle(self, positions: dict, equity: float, day: dict, bar_now: int) -> bool:
        """Returns True once every symbol has processed the bar `bar_now`."""
        open_trades = {t["symbol"]: t for t in self.st.open_trades(self.m.mode)}
        done = True
        refused = {}
        for sym in self.m.symbols:
            sig = self.signal_for(sym)
            if not sig or sig["bar_ts"] < bar_now:
                done = False  # exchange hasn't published the candle yet; retry next tick
            if not sig or sig["bar_ts"] == self.st.get(f"bar:{sym}"):
                continue
            self.st.set(f"bar:{sym}", sig["bar_ts"])
            regime = {1: "trend", -1: "range", 0: "дунд"}[sig["regime"]]
            log.info("%s оноо=%+.2f дохио=%s regime=%s%s %s", sym, sig["score"], sig["signal"], regime,
                     " (хэт хэлбэлзэлтэй)" if sig["volatile"] else "", sig["parts"])
            t = open_trades.get(sym)
            if t and sym in positions:
                if self.manage_open(t, sig, positions):
                    positions.pop(sym, None)
                continue
            if not self.m.is_open():  # e.g. FX at the weekend: manage open trades, open nothing new
                continue
            if sig["signal"] != 0 and sym not in positions and not self.st.get(f"pending:{sym}"):
                why = self.try_enter(sym, sig, positions, equity, day)
                if why:
                    refused.setdefault(why, []).append((sym, sig["signal"], sig["score"]))
        for why, items in refused.items():  # one message per reason; the same list isn't repeated for 3h
            key = "skip:" + why + ":" + ",".join(s for s, _, _ in items)
            notify(M.skipped(items, why), key=key, every_sec=3 * 3600)
        return done

    # ================= Telegram =================
    def status_text(self, equity: float) -> str:
        day = self.st.get("day") or {}
        dop = self.st.get("dopamine", 0.0)
        start = day.get("start_equity") or equity
        trades = self.st.open_trades(self.m.mode)
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
        pend = [(s, self.st.get(f"pending:{s}")) for s in self.m.symbols]
        pend = [(s, p) for s, p in pend if p]
        if pend:
            lines += ["", "━━━ ХҮЛЭЭГДЭЖ БУЙ ЗАХИАЛГА ━━━"]
            lines += [f"🎯 {base_asset(s)} {SIDE_NAME[p['side']].upper()} limit @ {M.px(p['limit'])}" for s, p in pend]
        carries = self.st.open_carries(self.m.mode)
        if carries:
            lines += ["", "━━━ FUNDING CARRY ━━━"]
            lines += [f"🏦 {base_asset(c['symbol'])}: funding {c['funding'] or 0:+.2f} $" for c in carries]
        if not trades and not pend:
            lines += ["", "Нээлттэй position алга. Дохио хүлээж байна."]
        return "\n".join(lines)

    def handle_commands(self, equity: float):
        for cmd in telegram_bot.poll(self.st):
            if cmd == "/status":
                notify(self.status_text(equity))
            elif cmd == "/stop":
                self.st.set("paused", True)
                notify("⏸ Шинэ арилжаа зогслоо. Нээлттэй position stop/tp-ээрээ үргэлжилнэ. /resume гэж эхлүүлнэ.")
            elif cmd == "/resume":
                self.st.set("paused", False)
                notify("▶️ Дахин эхэллээ.")
            elif cmd == "/closeall":
                self.st.set("closeall_asked", now_ms())
                notify("Бүх position (carry орно) хаах уу? 2 минутын дотор /closeall_yes гэж бичнэ үү.")
            elif cmd == "/closeall_yes":
                if now_ms() - self.st.get("closeall_asked", 0) < 120_000:
                    self.close_everything("manual", include_carry=True)
                    self.st.set("paused", True)
                    notify("🧹 Бүгдийг хаалаа, бот зогссон төлөвт байна (/resume).")
                else:
                    notify("Хугацаа дууссан. Эхлээд /closeall гэж бичнэ үү.")
            elif cmd == "/report":
                notify(review.weekly_report(self.st, self.m.mode))
            elif cmd == "/resetpeak":
                self.st.set("peak_equity", equity)
                notify(f"Оргил equity-г {equity:.2f} болгож шинэчиллээ. Drawdown хязгаар дахин тоологдоно.")
            else:
                notify(telegram_bot.HELP)

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

    def maybe_hourly(self, equity: float, day: dict):
        hour = now_ms() // 3_600_000
        if self.st.get("hourly_done") == hour:
            return
        self.st.set("hourly_done", hour)
        if hour % 6 == 0 or not self.st.get("calendar"):
            try:
                events.refresh(self.st)
            except Exception as e:
                log.warning("Календарь: %s", e)
        self.m.hourly(self, equity, day)  # market-specific hourly jobs (crypto: funding carry)

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
        rep = improver.improve(self.st, self.m.symbols, dopamine=self.st.get("dopamine", 0.0))
        for s in rep.get("new_shadows", []):
            notify(f"🧪 Шинэ shadow {s['id']}: vault оноо {s['vault']['score']}, бодит цаг хугацаанд туршиж эхэллээ")
        if rep.get("shadow_promoted"):
            notify(f"🧠 Shadow {rep['shadow_promoted']} ирээдүйн өгөгдөл дээр давж гарсан тул идэвхтэй тохиргоо боллоо")
        self.params = load_params()
        if now.weekday() == 0 and self.st.get("weekly_on") != today:
            self.st.set("weekly_on", today)
            notify(review.weekly_report(self.st, self.m.mode))

    # ================= main loop =================
    def tick(self):
        positions = self.broker.positions()
        equity = self.broker.equity()
        day = self.day_state(equity)
        self.peak(equity)
        if time.time() - self.last_equity_log > 300:
            self.last_equity_log = time.time()
            self.st.log_equity(self.m.mode, equity, self.st.get("dopamine", 0.0))
        self.handle_commands(equity)
        self.process_pending()
        positions = self.broker.positions()
        self.reconcile(self.directional(positions))
        self.kill_switch(equity, day)
        self.maybe_news()
        self.maybe_hourly(equity, day)
        positions = self.directional(self.broker.positions())
        bar_now = (now_ms() // self.tf_ms) * self.tf_ms - self.tf_ms  # last fully closed bar
        if self.st.get("checked_bar") != bar_now and self.on_new_candle(positions, equity, day, bar_now):
            self.st.set("checked_bar", bar_now)
        self.maybe_daily()

    def run(self):
        from . import dashboard
        keep_awake(True)
        try:
            dashboard.serve(background=True)
        except OSError as e:
            log.warning("Dashboard асаж чадсангүй: %s", e)
        notify(M.startup(self.m.mode, self.m.symbols, self.broker.equity()))
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
                pending = any(self.st.get(f"pending:{s}") for s in self.m.symbols)
                time.sleep(5 if pending else LOOP_SECONDS)
        finally:
            keep_awake(False)
