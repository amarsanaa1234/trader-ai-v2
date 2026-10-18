"""Telegram message layouts: a title line, then one fact per line."""
from .settings import base_asset
from .context import mkt

SIDE_WORD = {1: "LONG 📈", -1: "SHORT 📉"}
PART_LABEL = {
    "trend": "Trend", "momentum": "Momentum (RSI)", "mean_rev": "Буцах хандлага", "breakout": "Breakout",
    "funding": "Funding", "fng": "Fear & Greed", "macd": "MACD", "news": "Мэдээ", "crowd": "Олон нийт",
    "top": "Том арилжаачид", "oi": "Open interest", "patterns": "Chart pattern", "library": "Стратегийн сан",
    "htf": "4ц/1өдрийн trend",
}
REASON = {
    "tp": "🎯 TP-д хүрсэн", "stop": "❌ Stop-д хүрсэн", "stop_after_partial": "🔒 Хагас авсны дараа stop",
    "flip": "🔄 Дохио эсрэг эргэсэн", "time": "⌛ Хугацаа дууссан", "trail": "🔒 Хамгаалалтын stop",
    "kill_switch": "🛑 Kill switch", "manual": "✋ Таны командаар", "no_stop": "🚨 Stop тавьж чадаагүй",
}


def px(x: float) -> str:
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:,.4f}"


def usd(x: float) -> str:
    return f"{x:+,.2f} $"


def limit_placed(sym, side, limit, score, threshold) -> str:
    return "\n".join([
        "🎯 ЗАХИАЛГА ТАВЬЛАА",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        f"Limit үнэ:  {px(limit)}",
        f"Оноо:       {score:+.2f} (босго ±{threshold:.2f})",
        "⏳ Үнэ ирэхийг 1 цаг хүлээнэ, ирэхгүй бол цуцална",
    ])


def opened(sym, side, avg, qty, stop, tp, partial, risk_fraction, parts: dict, weights: dict) -> str:
    dist = abs(avg - stop)
    half = qty / 2 if partial else 0.0
    gain_partial = half * abs(partial - avg) if partial else 0.0
    gain_tp = (qty - half) * abs(tp - avg)
    lines = [
        "🚀 АРИЛЖАА НЭЭГДЛЭЭ",
        f"{base_asset(sym)} {SIDE_WORD[side]} @ {px(avg)}",
        f"Хэмжээ: {qty:g} {base_asset(sym)} (≈ {qty * avg:,.0f} $)",
        "",
        f"❌ Stop:   {px(stop)}  ({usd(-qty * dist)} · {dist / avg:.1%})",
    ]
    if partial:
        lines.append(f"💰 Хагас:  {px(partial)}  ({usd(gain_partial)})")
    lines += [f"🎯 TP:     {px(tp)}  ({usd(gain_tp)})",
              f"Эрсдэл: {risk_fraction:.2%} · Ашиг/эрсдэл ≈ {(gain_partial + gain_tp) / (qty * dist):.1f}",
              "", "📌 Гол шалтгаан:"]
    contrib = {k: v * weights.get(k, 0.0) for k, v in parts.items()}
    for k in sorted(contrib, key=lambda k: -abs(contrib[k]))[:4]:
        if abs(contrib[k]) < 0.01:
            continue
        mark = "✅" if contrib[k] * side > 0 else "⚠️"
        lines.append(f"  {mark} {PART_LABEL.get(k, k)} {parts[k]:+.2f}")
    lines.append("  (✅ дэмжсэн · ⚠️ эсэргүүцсэн)")
    return "\n".join(lines)


def closed(sym, side, entry, exit_px, pnl, r, reason, hours, rew, dop, mood) -> str:
    return "\n".join([
        f"{'✅' if pnl > 0 else '🔻'} АРИЛЖАА ХААГДЛАА",
        f"{base_asset(sym)} {SIDE_WORD[side]} · {hours:.0f} цаг",
        "",
        f"Шалтгаан:  {REASON.get(reason, reason)}",
        f"Орсон:     {px(entry)}",
        f"Гарсан:    {px(exit_px)}",
        f"Дүн:       {usd(pnl)} ({r:+.2f}R)",
        "",
        f"🧠 Шагнал {rew:+.2f} · Допамин {dop:+.2f} {mood}",
    ])


def partial_taken(sym, side) -> str:
    return "\n".join([
        "💰 ХАГАС АШИГ АВЛАА",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        "Position-ий хагасыг ашигтай хаалаа.",
        "Үлдсэн хагас нь TP хүртэл үргэлжилнэ.",
    ])


def stop_moved(sym, side, old, new, entry, qty) -> str:
    worst = side * qty * (new - entry)
    return "\n".join([
        "🔒 STOP ШИЛЖЛЭЭ",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        f"Хуучин:  {px(old)}",
        f"Шинэ:    {px(new)}",
        f"Хамгийн муу тохиолдолд: {usd(worst)}",
    ])


def skipped(items: list, reason: str) -> str:
    lines = ["⏸ ДОХИО АЛГАССАН", f"Шалтгаан: {reason}", ""]
    lines += [f"• {base_asset(s)} {SIDE_WORD[side]} (оноо {score:+.2f})" for s, side, score in items]
    return "\n".join(lines)


def daily(date, change, change_usd, equity, dop, mood, closed_today: int) -> str:
    return "\n".join([
        f"📊 ӨДРИЙН ДҮН · {date}",
        "",
        f"Дүн:        {change:+.2%} ({usd(change_usd)})",
        f"Данс:       {equity:,.2f} {mkt().currency}",
        f"Хаагдсан:   {closed_today} арилжаа",
        f"🧠 Допамин: {dop:+.2f} {mood}",
    ])


def startup(mode, symbols, equity) -> str:
    return "\n".join([
        "🤖 БОТ АСЛАА",
        "",
        f"Горим:  {mode}",
        f"Койн:   {', '.join(base_asset(s) for s in symbols)}",
        f"Данс:   {equity:,.2f} {mkt().currency}",
        "/status · /help",
    ])


def kill_switch(limit) -> str:
    return "\n".join([
        "🛑 KILL SWITCH",
        "",
        f"Өдрийн алдагдал {limit:.0%}-д хүрлээ.",
        "Бүх position хаагдлаа.",
        "Маргааш (UTC) хүртэл шинэ арилжаа нээхгүй.",
    ])


def review(trade_id, sym, side, lesson) -> str:
    return "\n".join([f"📝 AI ДҮГНЭЛТ · #{trade_id}", f"{base_asset(sym)} {SIDE_WORD[side]}", "", lesson])
