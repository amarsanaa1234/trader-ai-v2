"""Brain 2: a paper bot on 1h candles, for comparison only (owner, 2026-10-27).

It never places an order and never touches anyone's account. One small service
(`python main.py brain2`) trades every market on paper as if it were live: when
an hour closes it fetches the new 1h candles, takes the decisions on them with
the live settings of that market (the same file, so it follows every change)
and moves its own positions on: entries, stops, half taken, take-profit, time
limit, the daily loss limit - the same rules and the same shared-account
simulation as the backtests (core/portfolio.py), carried on from where it stopped,
so a trade it took is never rewritten later.

The bigger-timeframe trend on 1h candles comes from 4h and 1 day (config "htf").
Max hold is kept in days: 42 four-hour candles become 168 one-hour candles.

Beside it the same paper bot runs on the market's own timeframe (4h) from the same
day, so the two are compared like for like; the dashboard also shows the real
account. Both started 30 days back (owner) at 10,000.

State per market and timeframe: "brain2_<tf>" in the market's database.
"""
import logging
import os
import time

import numpy as np

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
SMALL_TF = "1h"
START_DAYS_BACK = 30         # owner, 2026-10-27: start 30 days back
WARMUP_DAYS = 120            # indicators and the 1-day trend need history before the start
START_EQUITY = 10_000.0
KEEP_TRADES = 300
KEEP_CURVE = 2000
log = logging.getLogger(__name__)


def own_tf(m) -> str:
    return str(m.cfg.get("timeframe", "4h"))


def key(tf: str) -> str:
    return f"brain2_{tf}"


def set_tf(m, tf: str) -> None:
    """Market.timeframe reads <MARKET>_TIMEFRAME; this process only runs brain 2."""
    os.environ[f"{m.name.upper()}_TIMEFRAME"] = tf


def sync(m, st) -> None:
    """New 1h candles of the traded instruments and the ones their sources read."""
    crypto = m.name == "crypto"
    syms = list(m.symbols)
    if not crypto:
        syms += [s for s in list(getattr(m, "aux_symbols", [])) + list(m.cfg.get("usd_basket") or [])
                 if s not in syms]
    for s in syms:
        try:
            if crypto:
                m.sync_symbol(st, s)
            else:
                from markets.fx import data
                data.sync_candles(st, s, SMALL_TF, m.history_days)
        except Exception as e:
            log.warning("brain2 %s %s: %s", m.name, s, e)


def step(m, st, tf: str, now: int | None = None) -> dict | None:
    """Carry the paper bot of market m on timeframe tf on to the last closed candle."""
    from . import portfolio
    from .features import add_static_features
    from .library import approved
    from .storage import now_ms
    from .strategy import clamp_params, load_params

    set_tf(m, tf)
    now = now or now_ms()
    tf_ms = m.tf_ms()
    last_closed = now // tf_ms * tf_ms - tf_ms
    b = st.get(key(tf))
    if b is None:
        start = (now // DAY_MS - START_DAYS_BACK) * DAY_MS
        b = {"tf": tf, "start": start, "done_ts": start - tf_ms, "state": None, "trades": [], "curve": [],
             "max_dd": 0.0, "n_trades": 0, "sum_r": 0.0}
    if b["done_ts"] >= last_closed:
        return b
    p = load_params()
    # the live settings count max hold in candles of the market's own timeframe: keep the days
    p = clamp_params({**p, "max_hold_bars": int(p["max_hold_bars"] * m.tf_ms(own_tf(m)) // tf_ms)})
    strats = approved()
    frames = {}
    for s in m.symbols:
        df = m.load_frame(st, s)
        df = df[(df.index >= b["done_ts"] - WARMUP_DAYS * DAY_MS) & (df.index <= last_closed)]
        if len(df) > 50:
            frames[s] = add_static_features(df, strats)
    if not frames:
        return b
    state = b["state"] or {"equity": START_EQUITY, "peak": START_EQUITY, "day": None, "day_start": START_EQUITY,
                           "halted": False, "trades_today": 0, "last_close": {}, "pos": {}, "pend": {}}
    r = portfolio.run(frames, p, start_ts=b["done_ts"] + 1, end_ts=last_closed + 1, with_curve=True, resume=state)
    if not len(r["times"]):
        return b
    b["state"], b["done_ts"] = r["state"], int(r["times"][-1])
    b["curve"] += [[int(t), round(float(e) / START_EQUITY - 1, 5)] for t, e in zip(r["times"], r["curve"])]
    if len(b["curve"]) > KEEP_CURVE:
        b["curve"] = b["curve"][::2]
    eq = np.array([1 + v for _, v in b["curve"]])
    b["max_dd"] = round(float(np.max(1 - eq / np.maximum.accumulate(eq))), 4)
    for t in r["trade_log"]:
        b["trades"].append({"symbol": t["symbol"], "ts": int(t["ts"]), "side": "long" if t["side"] > 0 else "short",
                            "r": round(float(t["r"]), 2), "pnl": round(float(t["pnl"]), 2), "reason": t["reason"]})
        b["n_trades"] += 1
        b["sum_r"] += float(t["r"])
    b["trades"] = b["trades"][-KEEP_TRADES:]
    last = {s: float(f["close"].iloc[-1]) for s, f in frames.items()}
    unreal = sum(q["side"] * q["qty"] * (last.get(s, q["entry"]) - q["entry"]) for s, q in b["state"]["pos"].items())
    b["equity"] = round(b["state"]["equity"] + unreal, 2)
    b["return"] = round(b["equity"] / START_EQUITY - 1, 5)
    b["avg_r"] = round(b["sum_r"] / b["n_trades"], 3) if b["n_trades"] else 0.0
    b["open"] = [{"symbol": s, "side": "long" if q["side"] > 0 else "short", "entry": q["entry"], "stop": q["stop"],
                  "tp": q["tp"], "price": last.get(s), "opened": q["bar_ts"],
                  "r_now": round(q["side"] * (last.get(s, q["entry"]) - q["entry"]) / q["dist"], 2)}
                 for s, q in b["state"]["pos"].items()]
    b["max_hold_bars"], b["updated"] = p["max_hold_bars"], now_ms()
    st.set(key(tf), b)
    return b


def tick() -> None:
    import markets
    from .context import activate
    from .storage import Storage
    for name in ("crypto", "fx", "commodities"):
        try:
            m = markets.get(name)
            activate(m)
            st = Storage()
            set_tf(m, SMALL_TF)
            sync(m, st)
            for tf in (SMALL_TF, own_tf(m)):
                step(m, st, tf)
        except Exception:
            log.exception("brain2 %s", name)


def run_forever() -> None:
    """Once per hour, a few minutes after the hour so the new candles exist."""
    done = None
    while True:
        hour = int(time.time() * 1000) // HOUR_MS
        if hour != done and time.time() % 3600 >= 120:
            tick()
            done = hour
        time.sleep(30)
