"""What the web pages show, for the ACTIVE market (core.web switches markets).

data(user)     everything the market page needs, for one person's account
summary(user)  a small card for the home page
public()       no money amounts: for the public performance page
"""
import json
import logging
import time

from . import events, library, reward, shadow
from .context import mkt
from .settings import BLACKOUT_AFTER_MIN, BLACKOUT_BEFORE_MIN, SHADOW_MIN_DAYS
from .storage import Storage, now_ms

log = logging.getLogger(__name__)
_prices = {}


def drift(st: Storage, user: str | None) -> dict:
    """Live results vs what the backtest promised (vault)."""
    trades = st.closed_trades(mkt().mode, 200, user)
    live_r = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    promised = None
    if mkt().params_path.exists():
        m = json.loads(mkt().params_path.read_text(encoding="utf-8")).get("metrics", {})
        promised = (m.get("vault") or m.get("backtest", {}).get("vault") or {}).get("avg_r")
    live = round(sum(live_r) / len(live_r), 3) if live_r else None
    warn = bool(live is not None and promised is not None and len(live_r) >= 15 and live < promised - 0.3)
    return {"live_avg_r": live, "live_trades": len(live_r), "backtest_avg_r": promised, "warning": warn}


def prices(symbols) -> dict:
    """Last traded prices, cached 15 s per market so many viewers don't hammer the exchange."""
    cache = _prices.setdefault(mkt().name, {"ts": 0.0, "data": {}})
    if symbols and time.time() - cache["ts"] > 15:
        try:
            cache["data"] = mkt().prices(symbols)
            cache["ts"] = time.time()
        except Exception as e:
            log.debug("prices: %s", e)
    return cache["data"]


def _account(st: Storage, user: str) -> dict:
    g = lambda k, d=None: st.get(f"{user}:{k}", d)   # noqa: E731  per-account state
    mode = mkt().mode
    hist = st.equity_history(mode, now_ms() - 30 * 86_400_000, user)
    eq = hist[-1]["equity"] if hist else None
    peak, dop = g("peak_equity"), g("dopamine", 0.0)
    pending = {r["key"].split(":pending:", 1)[1]: json.loads(r["value"]) for r in st.db.execute(
        "SELECT key, value FROM state WHERE key LIKE ?", (f"{user}:pending:%",))}
    return {"hist": hist, "equity": eq, "peak": peak, "dop": dop, "day": g("day"), "paused": g("paused", False),
            "pending": {k: v for k, v in pending.items() if v}}


def data(user: str) -> dict:
    st = Storage()
    m = mkt()
    mode, a = m.mode, _account(st, user)
    params = json.loads(m.params_path.read_text(encoding="utf-8")) if m.params_path.exists() else {}
    open_trades = st.open_trades(mode, user)
    step = max(1, len(a["hist"]) // 600)
    return {
        "market": m.name, "market_label": m.label, "user": user, "mode": mode, "symbols": m.symbols,
        "equity": a["equity"], "peak": a["peak"],
        "drawdown": round(1 - a["equity"] / a["peak"], 4) if a["equity"] and a["peak"] else 0.0,
        "dopamine": a["dop"], "mood": reward.mood(a["dop"]), "day": a["day"], "paused": a["paused"],
        "paper_started": st.get("paper_started"),
        "equity_history": a["hist"][::step],
        "open_trades": open_trades,
        "prices": prices({t["symbol"] for t in open_trades}),
        "pending": a["pending"],
        "trades": st.closed_trades(mode, 300, user),   # the page shows 10 at a time
        "heartbeat": st.get("heartbeat"),             # when the bot last ran (is it running?)
        "params": {"note": params.get("note"), "updated": params.get("updated_utc"), "metrics": params.get("metrics"),
                   "pattern_edges": params.get("params", {}).get("pattern_edges")},
        "shadows": [{**{k: s.get(k) for k in ("id", "created", "forward", "live_forward", "backtest")},
                     "days": round((now_ms() - s["created"]) / 86_400_000, 1)} for s in st.get("shadows", [])],
        "shadow_rules": {"min_days": SHADOW_MIN_DAYS, "min_trades": shadow.MIN_FORWARD_TRADES,
                         "margin": shadow.PROMOTE_MARGIN, "max_age_days": shadow.MAX_AGE_DAYS},
        "library": [{k: s.get(k) for k in ("key", "name", "status", "source", "description", "long_entry",
                                           "short_entry", "stats", "added", "reviewed")} for s in library.load()],
        "carry_open": st.open_carries(mode, user),
        "carry_closed": [dict(r) for r in st.db.execute(
            "SELECT * FROM carry WHERE status='closed' AND mode=? AND user=? ORDER BY closed_ts DESC LIMIT 20",
            (mode, user))],
        "events": events.upcoming(st, hours=24 * 7),
        "blackout": {"before_min": BLACKOUT_BEFORE_MIN, "after_min": BLACKOUT_AFTER_MIN, "now": events.blackout(st)},
        "drift": drift(st, user),
        **m.dashboard_extras(st),   # market-specific panels
    }


def summary(user: str) -> dict:
    """Home-page card: this person's account on the active market."""
    st = Storage()
    m = mkt()
    a = _account(st, user)
    day = a["day"] or {}
    start = day.get("start_equity")
    return {"market": m.name, "label": m.label, "mode": m.mode, "equity": a["equity"],
            "today": (a["equity"] / start - 1) if a["equity"] and start else None,
            "open": len(st.open_trades(m.mode, user)), "pending": len(a["pending"]), "paused": a["paused"],
            "halted": bool(day.get("halted")), "dopamine": a["dop"], "mood": reward.mood(a["dop"]),
            "heartbeat": st.get("heartbeat")}


def public(user: str) -> dict:
    """Performance without amounts (returns in %, counts), for showing other people."""
    st = Storage()
    m = mkt()
    hist = st.equity_history(m.mode, 0, user)
    trades = st.closed_trades(m.mode, 1000, user)
    base = hist[0]["equity"] if hist else None
    step = max(1, len(hist) // 300)
    wins = sum(1 for t in trades if (t["pnl"] or 0) > 0)
    return {"market": m.name, "label": m.label, "mode": m.mode, "since": hist[0]["ts"] if hist else None,
            "return": (hist[-1]["equity"] / base - 1) if base else None,
            "curve": [{"ts": h["ts"], "r": h["equity"] / base - 1} for h in hist[::step]] if base else [],
            "trades": len(trades), "win_rate": wins / len(trades) if trades else None,
            "avg_r": round(sum(t["r_multiple"] or 0 for t in trades) / len(trades), 3) if trades else None}
