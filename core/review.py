"""After-action reviews: each closed trade is explained from the numbers (why it
was opened, what the market did, which known mistakes apply), checked again
24h after a stop-out, and once a week summed up per person.

Written twice: English for Telegram (the owner is learning English) and
Mongolian, stored in the database for the dashboard. They do not change
decisions; the numbers (R, dopamine, walk-forward tests) do that."""
import json
import logging

import numpy as np

from . import ai
from .context import mkt
from .messages import PART_LABEL
from .storage import Storage, now_ms

log = logging.getLogger(__name__)

WEEK_SYSTEM = """You write a short weekly report, in simple English (the reader is learning English), for \
the owner of an automated {market} trading bot. Given the week's statistics and the repeated loss reasons, write \
3-5 bullet points: what worked, what didn't, any repeated pattern in the losses, and one thing to watch next \
week. Plain and honest; no hype, no investment advice."""
WEEK_SCHEMA = {"type": "object", "properties": {"bullets": {"type": "array", "items": {"type": "string"}}},
               "required": ["bullets"], "additionalProperties": False}

LABEL = {
    "mn": {"trend": "Trend", "momentum": "Momentum (RSI)", "mean_rev": "Буцах хандлага", "breakout": "Breakout",
           "funding": "Funding", "fng": "Fear & Greed", "macd": "MACD", "news": "Мэдээ", "crowd": "Олон нийт",
           "top": "Том арилжаачид", "oi": "Open interest", "patterns": "Chart pattern",
           "library": "Стратегийн сан", "htf": "Их хугацааны trend", "usd": "Долларын чиглэл",
           "tsmom": "Олон долоо хоногийн momentum", "carry": "Хүүгийн carry", "rates": "Хүүгийн өөрчлөлт",
           "cot": "Том фондууд (COT)", "xasset": "Нефть / алт / хувьцаа", "session": "Сессийн breakout",
           "orders": "Хүлээгдэж буй захиалга", "yields": "Бондын өгөөж", "month_end": "Сарын эцсийн урсгал"},
    "en": PART_LABEL,
}
HOUR = 3_600_000
FOLLOWUP_HOURS = 24


def _bars(st: Storage, symbol: str, start_ms: int, end_ms: int):
    df = mkt().load_frame(st, symbol)
    return df[(df.index >= start_ms) & (df.index < end_ms)]


def _short(symbol: str) -> str:
    return symbol.split("/")[0]


def analyze(st: Storage, t: dict) -> dict:
    """Numbers behind a closed trade: why it was opened, what the market did
    meanwhile. Pure arithmetic on stored data, no AI guessing."""
    from .strategy import load_params
    p = load_params()
    side = 1 if t["side"] == "long" else -1
    sig = json.loads(t.get("signal_json") or "{}")
    parts = sig.get("parts") or {}
    contrib = {k: v * p.get("w_range_mr" if k == "mean_rev" else f"w_{k}", 0.0) for k, v in parts.items()}
    support = sum(c * side for c in contrib.values() if c * side > 0)
    oppose = sum(abs(c) for c in contrib.values() if c * side < 0)
    dist = t.get("dist") or abs(t["entry"] - t["stop"]) or 1e-9

    bars = _bars(st, t["symbol"], t["opened_ts"] - HOUR, t["closed_ts"])
    mfe = mae = 0.0
    if len(bars):
        best = bars["high"].max() if side == 1 else bars["low"].min()
        worst = bars["low"].min() if side == 1 else bars["high"].max()
        mfe = max(0.0, side * (best - t["entry"]) / dist)
        mae = max(0.0, -side * (worst - t["entry"]) / dist)
    # the market's bellwether (crypto: BTC); markets without one skip this line
    ref = mkt().cfg.get("review_reference")
    ref_move = None
    if ref and not t["symbol"].startswith(_short(ref)):
        rb = _bars(st, ref, t["opened_ts"] - HOUR, t["closed_ts"])
        ref_move = float(rb["close"].iloc[-1] / rb["open"].iloc[0] - 1) if len(rb) else None
    others = st.db.execute(
        "SELECT symbol FROM trades WHERE mode=? AND id!=? AND side=? AND opened_ts<=? AND COALESCE(closed_ts, ?)>=? "
        "AND (? IS NULL OR user=?)",
        (t["mode"], t["id"], t["side"], t["closed_ts"], now_ms(), t["opened_ts"], t.get("user"), t.get("user"))
    ).fetchall()
    return {"side": side, "parts": parts, "contrib": contrib,
            "oppose_share": oppose / (support + oppose) if support + oppose else 0.0,
            "margin": abs(sig.get("score", 0)) - p["entry_threshold"], "threshold": p["entry_threshold"],
            "score": sig.get("score", 0), "regime": sig.get("regime"), "htf": parts.get("htf", 0.0),
            "mfe": mfe, "mae": mae, "ref": _short(ref) if ref else None, "ref_move": ref_move,
            "hours": (t["closed_ts"] - t["opened_ts"]) / HOUR,
            "same_side": sorted({_short(o["symbol"]) for o in others})}


def lessons(t: dict, a: dict, lang: str = "en") -> list[str]:
    """Plain-language mistakes (losses) or strengths (wins), most important first."""
    en = lang == "en"
    side, out, names = a["side"], [], ", ".join(a["same_side"])
    ref, move = a["ref"], a["ref_move"]
    if (t["pnl"] or 0) <= 0:
        if a["mfe"] < 0.3:
            out.append(f"It went against us right after entry (best moment only +{a['mfe']:.1f}R). "
                       "The direction or the entry timing was wrong." if en else
                       f"Орсон даруйдаа эсрэг тийш явсан (хамгийн их ашиг ердөө +{a['mfe']:.1f}R). "
                       "Чиглэл эсвэл орох цаг буруу байсан.")
        elif a["mfe"] >= 0.8:
            out.append(f"It first moved our way to +{a['mfe']:.1f}R, then turned back. The profit was not protected in time."
                       if en else f"Эхэндээ зөв явж +{a['mfe']:.1f}R хүрсэн ч буцаж эргэсэн. Ашгаа хамгаалж амжаагүй.")
        if len(a["same_side"]) >= 2:
            out.append(f"At the same time we were also {t['side']} on {names}. These move together, so when the "
                       "market turned they all lost together (one big bet)." if en else
                       f"Тэр үед {names} дээр мөн {t['side']} байсан. Бүгд хамт хөдөлдөг тул "
                       "зах зээл эргэхэд хамт алдсан (нэг том бооцоо болсон).")
        if move is not None and move * side < -0.01:
            out.append(f"{ref} moved {move:+.1%} during the trade: the whole market went against us." if en else
                       f"Арилжааны хугацаанд {ref} {move:+.1%} хөдөлсөн. Бүх зах зээл эсрэг тийш явсан.")
        if a["margin"] < 0.15:
            out.append(f"Weak signal: the score was only {a['margin']:.2f} above the threshold." if en else
                       f"Дохио сул байсан: оноо босгоос ердөө {a['margin']:.2f}-оор давсан.")
        if a["oppose_share"] > 0.3:
            out.append(f"Mixed signals: {a['oppose_share']:.0%} of the signal strength pointed the other way." if en else
                       f"Дохионууд зөрчилтэй байсан: эсэргүүцсэн дохионы хүч {a['oppose_share']:.0%}.")
        if a["htf"] * side < -0.3:
            out.append("We traded against the bigger-timeframe trend." if en else "Их хугацааны trend-ийн эсрэг орсон.")
        if a["regime"] == -1:
            out.append("The market was moving sideways (range). Trend signals work badly then." if en else
                       "Хажуу тийш (range) зах зээлд орсон. Энэ үед trend-ийн дохио муу ажилладаг.")
        if a["hours"] <= 3 and a["mae"] >= 0.9:
            out.append(f"The stop was hit within {a['hours']:.0f} hours: a sudden sharp move (news?)." if en else
                       f"Ердөө {a['hours']:.0f} цагийн дотор stop-д хүрсэн. Гэнэтийн огцом хөдөлгөөн (мэдээ?) байсан байж магадгүй.")
        if not out:
            out.append("No clear mistake: the signal was good enough and the rules were followed. "
                       "This is a normal, unavoidable loss." if en else
                       "Тодорхой алдаа олдсонгүй: дохио хангалттай, дүрмээ баримталсан. "
                       "Энэ бол ердийн, зайлшгүй тохиолдох алдагдал.")
    else:
        strength = ("strong" if en else "хүчтэй") if a["margin"] >= 0.3 else ("medium" if en else "дунд зэрэг")
        out.append(f"The signal was {strength} (+{a['margin']:.2f} above the threshold)." if en else
                   f"Дохио {strength} байсан (босгоос +{a['margin']:.2f}).")
        if a["oppose_share"] < 0.2:
            out.append("The signals agreed with each other." if en else "Дохионууд нэг чиглэлд санал нийлсэн.")
        if a["htf"] * side > 0.3:
            out.append("We traded with the bigger trend." if en else "Том trend-ийн чиглэлд орсон.")
        if move is not None and move * side > 0.01:
            out.append(f"The whole market moved our way ({ref} {move:+.1%})." if en else
                       f"Зах зээл бүхэлдээ ижил чиглэлд явсан ({ref} {move:+.1%}).")
    return out


def render(t: dict, a: dict, lang: str = "en") -> str:
    en, side, lab = lang == "en", a["side"], LABEL[lang]
    top = sorted(a["contrib"].items(), key=lambda kv: -abs(kv[1]))
    sup = [f"{lab.get(k, k)} {a['parts'][k]:+.2f}" for k, c in top if c * side > 0][:3]
    opp = [f"{lab.get(k, k)} {a['parts'][k]:+.2f}" for k, c in top if c * side < 0][:2]
    regime = ({1: "trending", -1: "sideways (range)", 0: "in between"} if en else
              {1: "trend", -1: "хажуу тийш (range)", 0: "дунд"}).get(a["regime"], "?")
    if en:
        lines = ["🎯 Why we entered:",
                 f"• Score {a['score']:+.2f} (threshold ±{a['threshold']:.2f}, margin {a['margin']:+.2f})",
                 f"• Supported by: {', '.join(sup) or '—'}"]
        if opp:
            lines.append(f"• Against: {', '.join(opp)} ({a['oppose_share']:.0%} of the strength)")
        lines += [f"• Market: {regime}", "", "📉 What happened:",
                  f"• Lasted {a['hours']:.0f} h, closed at {t['r_multiple']:+.2f}R",
                  f"• Best moment +{a['mfe']:.1f}R, worst moment −{a['mae']:.1f}R"]
        if a["ref_move"] is not None:
            lines.append(f"• {a['ref']} moved {a['ref_move']:+.1%} meanwhile")
        if a["same_side"]:
            lines.append(f"• Also {t['side']} at the same time: {', '.join(a['same_side'])}")
        lines += ["", "🧠 Lessons:"]
    else:
        lines = ["🎯 Яагаад орсон бэ:",
                 f"• Оноо {a['score']:+.2f} (босго ±{a['threshold']:.2f}, зөрүү {a['margin']:+.2f})",
                 f"• Дэмжсэн: {', '.join(sup) or '—'}"]
        if opp:
            lines.append(f"• Эсэргүүцсэн: {', '.join(opp)} (эсрэг хүч {a['oppose_share']:.0%})")
        lines += [f"• Зах зээл: {regime}", "", "📉 Юу болсон бэ:",
                  f"• {a['hours']:.0f} цаг үргэлжилж {t['r_multiple']:+.2f}R-ээр хаагдсан",
                  f"• Хамгийн ашигтай үе +{a['mfe']:.1f}R, хамгийн муу үе −{a['mae']:.1f}R"]
        if a["ref_move"] is not None:
            lines.append(f"• Тэр хугацаанд {a['ref']} {a['ref_move']:+.1%}")
        if a["same_side"]:
            lines.append(f"• Зэрэг {t['side']} байсан: {', '.join(a['same_side'])}")
        lines += ["", "🧠 Дүгнэлт / сургамж:"]
    return "\n".join(lines + [f"• {x}" for x in lessons(t, a, lang)])


def review_trade(st: Storage, trade_id: int) -> str | None:
    """Stores the Mongolian review (dashboard), returns the English one (Telegram)."""
    t = dict(st.db.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone())
    try:
        a = analyze(st, t)
        st.update_trade(trade_id, review=render(t, a, "mn"))
        return render(t, a, "en")
    except Exception as e:
        log.warning("Trade review #%s: %s", trade_id, e)
        return None


def followups(st: Storage) -> list[tuple]:
    """24h after a stop-out: did price come back to the target (stop too tight)
    or keep going (the stop saved money)? The Mongolian note is appended to the
    stored review once; (id, symbol, side, user, English note) is returned for Telegram."""
    done = []
    rows = st.db.execute(
        "SELECT * FROM trades WHERE status='closed' AND exit_reason IN ('stop', 'stop_after_partial') "
        "AND closed_ts <= ? AND closed_ts >= ? AND review LIKE '%🎯%' AND review NOT LIKE '%⏰%'",
        (now_ms() - FOLLOWUP_HOURS * HOUR, now_ms() - 7 * 86_400_000)).fetchall()
    for r in rows:
        t = dict(r)
        side = 1 if t["side"] == "long" else -1
        dist = t.get("dist") or abs(t["entry"] - t["stop"]) or 1e-9
        after = _bars(st, t["symbol"], t["closed_ts"], t["closed_ts"] + FOLLOWUP_HOURS * HOUR)
        expected = FOLLOWUP_HOURS * HOUR // mkt().tf_ms()
        if len(after) < max(1, expected - 2):
            continue
        best = after["high"].max() if side == 1 else after["low"].min()
        worst = after["low"].min() if side == 1 else after["high"].max()
        further = -side * (worst - t["exit"]) / dist
        if t["tp"] and side * (best - t["tp"]) >= 0:
            mn = "⏰ 24 цагийн дараа: үнэ буцаж эргээд TP хүртэл явсан. Чиглэл зөв байсан ч stop хэт ойр байсан."
            en = "24 hours later the price turned around and reached the target. The direction was right, but the stop was too close."
        elif further >= 1.0:
            mn = f"⏰ 24 цагийн дараа: үнэ цааш {further:.1f}R эсрэг явсан. Stop зөв ажиллаж, илүү том алдагдлаас хамгаалсан ✅"
            en = f"24 hours later the price had moved another {further:.1f}R against us. The stop worked and saved us from a bigger loss ✅"
        else:
            mn = "⏰ 24 цагийн дараа: үнэ тодорхой чиглэлгүй хэлбэлзсэн. Stop-ын шийдвэр хэвийн байсан."
            en = "24 hours later the price was just moving up and down. The stop decision was normal."
        st.update_trade(t["id"], review=t["review"] + "\n\n" + mn)
        done.append((t["id"], t["symbol"], t["side"], t.get("user"), en))
    return done


def _lesson_counts(trades) -> dict:
    """How often each kind of mistake showed up in the losing trades (read from the stored reviews)."""
    keys = {"эсрэг тийш явсан": "Went against us right away", "нэг том бооцоо": "All in one direction",
            "Бүх зах зээл эсрэг": "Whole market against us", "Дохио сул": "Weak signal",
            "зөрчилтэй": "Mixed signals", "trend-ийн эсрэг": "Against the bigger trend",
            "(range) зах зээлд орсон": "Sideways market", "stop хэт ойр": "Stop too close"}
    out = {}
    for t in trades:
        if (t["pnl"] or 0) <= 0 and t.get("review"):
            for k, name in keys.items():
                if k in t["review"]:
                    out[name] = out.get(name, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def weekly_stats(st: Storage, mode: str, user: str | None = None) -> dict:
    since = now_ms() - 7 * 86_400_000
    trades = [t for t in st.trades_since(mode, since, user) if t["status"] == "closed"]
    rs = [t["r_multiple"] or 0 for t in trades]
    pnl = sum(t["pnl"] or 0 for t in trades)
    eq = st.equity_history(mode, since, user)
    by_reason = {}
    for t in trades:
        by_reason[t["exit_reason"]] = by_reason.get(t["exit_reason"], 0) + 1
    return {"trades": len(trades), "wins": sum(1 for t in trades if (t["pnl"] or 0) > 0),
            "pnl": round(pnl, 2), "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0,
            "equity_change": round(eq[-1]["equity"] / eq[0]["equity"] - 1, 4) if len(eq) > 1 else 0.0,
            "exits": by_reason, "loss_reasons": _lesson_counts(trades)}


def weekly_report(st: Storage, mode: str, user: str | None = None) -> str:
    """English, for Telegram."""
    s = weekly_stats(st, mode, user)
    head = (f"📅 7-DAY REPORT · {mkt().label}\n\n{s['trades']} trades, {s['wins']} wins, "
            f"{s['pnl']:+.2f} {mkt().currency}, average {s['avg_r']:+.2f}R, balance {s['equity_change']:+.2%}")
    if s["loss_reasons"]:
        head += "\n🧠 Repeated loss reasons: " + ", ".join(f"{k} ×{v}" for k, v in s["loss_reasons"].items())
    if not ai.enabled() or s["trades"] == 0:
        return head
    try:
        ans = ai.ask_json(WEEK_SYSTEM.format(market=mkt().name), json.dumps(s, ensure_ascii=False), WEEK_SCHEMA)
        if ans and ans.get("bullets"):
            return head + "\n\n" + "\n".join(f"• {b}" for b in ans["bullets"][:5])
    except Exception as e:
        log.warning("Weekly report: %s", e)
    return head
