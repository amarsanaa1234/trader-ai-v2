"""After-action reviews: the AI explains each closed trade in one or two
sentences, and once a week sums up what the bot did and learned.

These are explanations for you. They do not change decisions; the numbers
(R, dopamine, walk-forward tests) do that."""
import json
import logging

import numpy as np

from . import ai
from .settings import REVIEW_LANG
from .storage import Storage, now_ms

log = logging.getLogger(__name__)

TRADE_SYSTEM = """You review a single closed crypto futures trade made by an automated bot. Explain in \
1-2 short sentences, in {lang}, the most likely reason it won or lost, based only on the \
data given (signal components at entry, regime, exit reason, R multiple). Be concrete and honest; a loss \
that hit its stop according to plan is not a mistake. Do not give investment advice."""
TRADE_SCHEMA = {"type": "object", "properties": {"lesson": {"type": "string"},
                                                  "tags": {"type": "array", "items": {"type": "string"}}},
                "required": ["lesson", "tags"], "additionalProperties": False}

WEEK_SYSTEM = """You write a short weekly report, in {lang}, for the owner of an automated \
crypto trading bot. Given the week's statistics and per-trade notes, write 3-5 bullet points: what worked, \
what didn't, any repeated pattern in the losses, and one thing to watch next week. Plain and honest; no \
hype, no investment advice."""
WEEK_SCHEMA = {"type": "object", "properties": {"bullets": {"type": "array", "items": {"type": "string"}}},
               "required": ["bullets"], "additionalProperties": False}


def review_trade(st: Storage, trade_id: int) -> str | None:
    if not ai.enabled():
        return None
    t = dict(st.db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone())
    sig = json.loads(t.get("signal_json") or "{}")
    facts = {k: t[k] for k in ("symbol", "side", "entry", "exit", "stop", "tp", "r_multiple", "exit_reason",
                               "partial_done", "regime")}
    # spell the outcome out; small local models misread the raw numbers
    facts["outcome"] = "WIN" if (t["pnl"] or 0) > 0 else "LOSS"
    facts["hours_held"] = round((t["closed_ts"] - t["opened_ts"]) / 3_600_000, 1)
    facts["signal_score"] = sig.get("score")
    facts["signal_components"] = sig.get("parts")
    try:
        ans = ai.ask_json(TRADE_SYSTEM.format(lang=REVIEW_LANG), json.dumps(facts, ensure_ascii=False), TRADE_SCHEMA)
    except Exception as e:
        log.warning("Арилжааны дүгнэлт: %s", e)
        return None
    if not ans:
        return None
    lesson = str(ans.get("lesson", ""))[:400]
    st.update_trade(trade_id, review=lesson)
    return lesson


def weekly_stats(st: Storage, mode: str) -> dict:
    since = now_ms() - 7 * 86_400_000
    trades = [t for t in st.trades_since(mode, since) if t["status"] == "closed"]
    rs = [t["r_multiple"] or 0 for t in trades]
    pnl = sum(t["pnl"] or 0 for t in trades)
    eq = st.equity_history(mode, since)
    by_reason = {}
    for t in trades:
        by_reason[t["exit_reason"]] = by_reason.get(t["exit_reason"], 0) + 1
    return {"trades": len(trades), "wins": sum(1 for t in trades if (t["pnl"] or 0) > 0),
            "pnl_usdt": round(pnl, 2), "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0,
            "equity_change": round(eq[-1]["equity"] / eq[0]["equity"] - 1, 4) if len(eq) > 1 else 0.0,
            "exits": by_reason, "notes": [t["review"] for t in trades if t.get("review")][:20]}


def weekly_report(st: Storage, mode: str) -> str:
    s = weekly_stats(st, mode)
    head = (f"📅 7 хоногийн тайлан: {s['trades']} арилжаа, ялалт {s['wins']}, {s['pnl_usdt']:+.2f} USDT, "
            f"дундаж {s['avg_r']:+.2f}R, equity {s['equity_change']:+.2%}")
    if not ai.enabled() or s["trades"] == 0:
        return head
    try:
        ans = ai.ask_json(WEEK_SYSTEM.format(lang=REVIEW_LANG), json.dumps(s, ensure_ascii=False), WEEK_SCHEMA)
        if ans and ans.get("bullets"):
            return head + "\n" + "\n".join(f"• {b}" for b in ans["bullets"][:5])
    except Exception as e:
        log.warning("Долоо хоногийн тайлан: %s", e)
    return head
