"""FX's main bot: the 15m learner on the OANDA demo account (owner, 2026-11-11: it replaced
the 4h FX bot there, 2% risk per trade; 2026-11-12: the 4h bot and FX brain 2 removed).
It also does the FX housekeeping the 4h bot used to do (calendar, headlines, the learner's
brain and AI helpers) - see housekeeping().

Brain and hands are separate. The brain is exactly the learner of brain 2
(core/orb_learn.py: same candidates, same beliefs, same lessons, AI lessons and safety
check from core/learn_ai.py), recomputed right after every 15m candle closes. When it
decides to take a trade that starts at the candle opening now, this module places it:

  entry   market order, stop attached on OANDA's side (2 x ATR, as decided)
  size    2% of the account x the learner's weight, at most MAX_TRADE_LEV x NAV notional
          per trade and MAX_TOTAL_LEV x NAV in total (OANDA retail margin is 30:1; a 2x ATR
          15m stop is ~0.1%, so 2% risk alone would ask for ~20x per trade)
  exit    the stop on OANDA, or a market close at the session's end (16:00 local), after
          64 candles, or Friday 20:00 UTC - the same exits as the learner
  limits  one position per pair (the account nets positions per pair); no new entry after
          -6% in a UTC day or -45% from the peak
The learner keeps learning from its own simulated trades (the same for every run); the
real fills are recorded beside them. State "fx15_live" in the FX database.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from . import learn_ai, orb15, orb_learn
from .notifier import notify
from .storage import Storage, now_ms

log = logging.getLogger(__name__)
BAR = orb15.BAR_MS
RISK = 0.02
MAX_TRADE_LEV, MAX_TOTAL_LEV = 12.0, 25.0
KEY = "fx15_live"
WAIT_AFTER_CLOSE_MS = 25_000       # OANDA publishes the closed candle a few seconds after the bar


def exit_due(pos: dict, now: int) -> str | None:
    """The learner's time exits for a live position."""
    s = orb_learn.SESSIONS[pos["session"]]
    t = pd.Timestamp(pos["entry_ts"], unit="ms", tz="UTC").tz_convert(s["tz"])
    flat = t.normalize() + pd.Timedelta(minutes=s["flat"])
    u = datetime.fromtimestamp(now / 1000, timezone.utc)
    if pd.Timestamp(now, unit="ms", tz="UTC") >= flat:
        return "session end"
    if now >= pos["entry_ts"] + orb15.P["hold"] * BAR:
        return "64 candles"
    if u.weekday() == 4 and u.hour >= 20:
        return "Friday close"
    return None


def decide(st, now: int) -> list[dict]:
    """The learner's trades that start at the candle opening now (weight > 0)."""
    import json
    snap = json.loads(orb_learn.snapshot_path().read_text(encoding="utf-8"))
    known, snap_end = snap["trades"], int(snap["end"])
    bar_open = now // BAR * BAR
    cands = []
    for pair in orb_learn.PAIRS:
        df = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf=? AND ts>=? "
                               "ORDER BY ts", st.db, params=(pair, orb15.TF, snap_end - orb15.WARMUP_DAYS * 86_400_000))
        df = df[df.ts < bar_open]                       # closed candles only
        if len(df) < 96 * 25:
            continue
        cands += [t for s in orb_learn.SESSIONS for t in orb_learn.candidates(df, pair, s, include_open=True)
                  if t["entry_ts"] > snap_end]
    book = orb_learn.replay(cands, snap_end, known, learn_ai.Context.load(st))["book"]
    return [t for t in book if t.get("open") and t["entry_ts"] == bar_open and t["weight"] > 0]


class Live:
    def __init__(self, st, broker, user):
        self.st, self.b, self.user = st, broker, user
        self.s = st.get(KEY) or {"positions": {}, "trades": [], "peak": None, "day": None, "day_start": None}

    def save(self):
        self.st.set(KEY, self.s)

    def record_close(self, pair: str, pos: dict, reason: str, exit_px=None, pnl=None):
        if pnl is None:
            exit_px, pnl = self.b.closed_result(pair, {"opened_ts": pos["entry_ts"]})
        risk = pos["risk"] or 1.0
        r = round(float(pnl or 0.0) / risk, 2)
        self.s["trades"] = (self.s["trades"] + [{**pos, "pair": pair, "exit_ts": now_ms(), "exit": exit_px,
                                                 "pnl": round(float(pnl or 0.0), 2), "r": r, "reason": reason}])[-300:]
        self.s["positions"].pop(pair, None)
        notify(f"{'✅' if (pnl or 0) > 0 else '🔻'} 15M TRADE CLOSED · {pair}\n{pos['side']} · "
               f"{orb_learn.SESSIONS[pos['session']]['tz'].split('/')[-1]} session\nResult: {r:+.2f}R "
               f"({float(pnl or 0):+.2f})\nReason: {reason}", to=self.user)

    def reconcile(self, now: int):
        """Positions closed on OANDA (stop hit) -> record; time exits -> close."""
        open_now = self.b.positions()
        for pair, pos in list(self.s["positions"].items()):
            if pair not in open_now:
                self.record_close(pair, pos, "stop")
                continue
            why = exit_due(pos, now)
            if why:
                px = self.b.close(pair)
                if pair not in self.b.positions():
                    self.record_close(pair, pos, why, exit_px=px)

    def enter(self, trades: list[dict], now: int):
        nav = self.b.equity()
        day = now // 86_400_000
        if self.s["day"] != day:
            self.s["day"], self.s["day_start"] = day, nav
        self.s["peak"] = max(self.s["peak"] or nav, nav)
        if nav / self.s["day_start"] - 1 <= -orb15.DAILY_LOSS or nav / self.s["peak"] - 1 <= -orb15.MAX_DD:
            return
        open_now = self.b.positions()
        used = sum(abs(p["qty"]) * p["entry"] for p in open_now.values())     # notional in account currency
        for t in sorted(trades, key=lambda t: -t["p_pos"]):
            pair = t["pair"]
            if pair in open_now or pair in self.s["positions"]:
                continue                                  # the account nets per pair: one at a time
            px = self.b.price(pair)
            qty = RISK * t["weight"] * nav / t["dist"]    # account-currency P&L per price unit
            qty = min(qty, MAX_TRADE_LEV * nav / px, max(0.0, MAX_TOTAL_LEV * nav - used) / px)
            ok_qty, why = self.b.check_size(pair, qty, px)
            if ok_qty <= 0:
                log.info("fx15 %s skipped: %s", pair, why)
                continue
            units = t["side"] * self.b._units(pair, qty)
            r = self.b._order({"type": "MARKET", "instrument": pair, "units": str(units), "timeInForce": "FOK",
                               "stopLossOnFill": {"distance": self.b._px(pair, t["dist"]), "timeInForce": "GTC"}})
            fill = r.get("orderFillTransaction")
            if not fill:
                log.warning("fx15 %s not filled: %s", pair, r.get("orderCancelTransaction", {}).get("reason"))
                continue
            entry = float(fill["price"])
            pos = {"arm": t["arm"], "session": t["session"], "side": "long" if t["side"] > 0 else "short",
                   "entry_ts": t["entry_ts"], "entry": entry, "stop": entry - t["side"] * t["dist"],
                   "qty": qty, "risk": qty * t["dist"], "belief": t["belief"], "p": t["p_pos"]}
            self.s["positions"][pair] = pos
            used += qty * entry
            notify(f"🟢 15M TRADE OPENED · {pair}\n{pos['side'].upper()} at {entry} · stop {pos['stop']:.5g}\n"
                   f"Arm: {t['session']} breakout · learner trusts it {t['p_pos']:.0%} (avg {t['belief']:+.2f}R)\n"
                   f"Risk: {pos['risk']:.2f} ({pos['risk'] / nav:.1%} of the account)", to=self.user)


def tick(st, broker, user, now: int | None = None):
    from markets.fx import data
    now = now or now_ms()
    for pair in orb_learn.PAIRS:
        data.sync_candles(st, pair, orb15.TF, orb15.START_DAYS_BACK + orb15.WARMUP_DAYS + 5)
    live = Live(st, broker, user)
    live.reconcile(now)
    live.enter(decide(st, now), now)
    nav = broker.equity()
    live.s["nav"], live.s["updated"] = nav, now
    live.s["equity"] = (live.s.get("equity") or [])[-3000:] + [[now, round(nav, 2)]]
    live.save()


def housekeeping(st, now: int, last: dict) -> None:
    """What the 4h FX bot used to do and the learner needs: the economic calendar (6-hourly),
    headlines and their AI scores (half-hourly), and the learner's brain with its AI helpers
    (hourly: beliefs, lessons, news risk, AI researcher - core/orb_learn.step)."""
    from . import events, news
    jobs = (("calendar", 6 * 3_600_000, lambda: events.refresh(st)),
            ("news", 30 * 60_000, lambda: (news.fetch_headlines(st), news.score_news(st))),
            ("brain", 3_600_000, lambda: orb_learn.step(st, now)))
    for name, every, job in jobs:
        if now - last.get(name, 0) >= every:
            last[name] = now
            try:
                job()
            except Exception as e:
                log.warning("fx15 %s: %s", name, e)


def run_forever():
    """Right after every 15m candle closes."""
    import markets
    from core import users
    from core.context import activate
    from markets.fx.broker import OandaBroker
    m = markets.get("fx")
    activate(m)
    st = Storage()
    user = next(u for u in users.load_users() if m.account_creds(u))
    acc = type("Acc", (), {"name": user.name, "creds": m.account_creds(user)})()
    broker = OandaBroker(st, acc)
    done, last = None, {}
    while True:
        bar = (now_ms() - WAIT_AFTER_CLOSE_MS) // BAR
        if bar != done:
            try:
                tick(st, broker, user)
            except Exception:
                log.exception("fx15 tick")
                notify("🚨 15m FX bot: a loop failed, retrying at the next candle (see the log)",
                       key="fx15fail", every_sec=3600, to=user)
            done = bar
            housekeeping(st, now_ms(), last)       # after the trades: AI calls can take minutes
        time.sleep(5)
