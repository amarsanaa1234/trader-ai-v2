"""Telegram message layouts (English, the owner is learning it): a title line,
then one fact per line."""
from .context import mkt
from .settings import base_asset

SIDE_WORD = {1: "LONG 📈", -1: "SHORT 📉"}
PART_LABEL = {
    "trend": "Trend", "momentum": "Momentum (RSI)", "mean_rev": "Mean reversion", "breakout": "Breakout",
    "funding": "Funding", "fng": "Fear & Greed", "macd": "MACD", "news": "News", "crowd": "Crowd positioning",
    "top": "Top traders", "oi": "Open interest", "patterns": "Chart patterns", "library": "Strategy library",
    "htf": "Bigger-timeframe trend",
    # FX
    "usd": "US dollar trend", "tsmom": "Multi-week momentum", "carry": "Interest-rate carry",
    "rates": "Interest-rate change", "cot": "Big funds (COT)", "xasset": "Oil / gold / stocks",
    "session": "Session breakout", "orders": "Waiting orders", "yields": "Bond yields", "month_end": "Month-end flows",
    # commodities
    "curve": "Futures curve (shortage / glut)", "curve_chg": "Futures curve change",
}
REASON = {
    "tp": "🎯 Take-profit hit", "stop": "❌ Stop-loss hit", "stop_after_partial": "🔒 Stop hit after taking half",
    "flip": "🔄 Signal flipped", "time": "⌛ Time limit reached", "trail": "🔒 Protective stop",
    "kill_switch": "🛑 Kill switch", "manual": "✋ Your command", "no_stop": "🚨 Could not place a stop",
    "weekend": "📅 Closed before the weekend",
}


def px(x: float) -> str:
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:,.4f}"


def usd(x: float) -> str:
    return f"{x:+,.2f} {mkt().currency}"


def limit_placed(sym, side, limit, score, threshold) -> str:
    wait_h = 2 * mkt().tf_ms() / 3_600_000
    return "\n".join([
        "🎯 ORDER PLACED",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        f"Limit price:  {px(limit)}",
        f"Score:        {score:+.2f} (threshold ±{threshold:.2f})",
        f"⏳ Waits up to {wait_h:.0f} hours for the price; cancelled if it doesn't come",
    ])


def opened(sym, side, avg, qty, stop, tp, partial, risk_fraction, parts: dict, weights: dict) -> str:
    dist = abs(avg - stop)
    half = qty / 2 if partial else 0.0
    gain_partial = half * abs(partial - avg) if partial else 0.0
    gain_tp = (qty - half) * abs(tp - avg)
    lines = [
        "🚀 TRADE OPENED",
        f"{base_asset(sym)} {SIDE_WORD[side]} @ {px(avg)}",
        f"Size: {qty:g} {base_asset(sym)} (≈ {qty * avg:,.0f} {mkt().currency})",
        "",
        f"❌ Stop:   {px(stop)}  ({usd(-qty * dist)} · {dist / avg:.1%})",
    ]
    if partial:
        lines.append(f"💰 Half:   {px(partial)}  ({usd(gain_partial)})")
    lines += [f"🎯 Target: {px(tp)}  ({usd(gain_tp)})",
              f"Risk: {risk_fraction:.2%} · Reward/risk ≈ {(gain_partial + gain_tp) / (qty * dist):.1f}",
              "", "📌 Main reasons:"]
    contrib = {k: v * weights.get(k, 0.0) for k, v in parts.items()}
    for k in sorted(contrib, key=lambda k: -abs(contrib[k]))[:4]:
        if abs(contrib[k]) < 0.01:
            continue
        mark = "✅" if contrib[k] * side > 0 else "⚠️"
        lines.append(f"  {mark} {PART_LABEL.get(k, k)} {parts[k]:+.2f}")
    lines.append("  (✅ supports · ⚠️ opposes)")
    return "\n".join(lines)


def closed(sym, side, entry, exit_px, pnl, r, reason, hours, rew, dop, mood) -> str:
    return "\n".join([
        f"{'✅' if pnl > 0 else '🔻'} TRADE CLOSED",
        f"{base_asset(sym)} {SIDE_WORD[side]} · {hours:.0f} h",
        "",
        f"Reason:  {REASON.get(reason, reason)}",
        f"Entry:   {px(entry)}",
        f"Exit:    {px(exit_px)}",
        f"Result:  {usd(pnl)} ({r:+.2f}R)",
        "",
        f"🧠 Reward {rew:+.2f} · Dopamine {dop:+.2f} {mood}",
    ])


def partial_taken(sym, side) -> str:
    return "\n".join([
        "💰 HALF PROFIT TAKEN",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        "Half of the position was closed in profit.",
        "The other half keeps running to the target.",
    ])


def stop_moved(sym, side, old, new, entry, qty) -> str:
    worst = side * qty * (new - entry)
    return "\n".join([
        "🔒 STOP MOVED",
        f"{base_asset(sym)} {SIDE_WORD[side]}",
        "",
        f"Old:  {px(old)}",
        f"New:  {px(new)}",
        f"Worst case now: {usd(worst)}",
    ])


def skipped(items: list, reason: str) -> str:
    lines = ["⏸ SIGNALS SKIPPED", f"Reason: {reason}", ""]
    lines += [f"• {base_asset(s)} {SIDE_WORD[side]} (score {score:+.2f})" for s, side, score in items]
    return "\n".join(lines)


def daily(date, change, change_usd, equity, dop, mood, closed_today: int) -> str:
    return "\n".join([
        f"📊 DAILY RESULT · {date}",
        "",
        f"Result:     {change:+.2%} ({usd(change_usd)})",
        f"Balance:    {equity:,.2f} {mkt().currency}",
        f"Closed:     {closed_today} trades",
        f"🧠 Dopamine: {dop:+.2f} {mood}",
    ])


def startup(mode, symbols, equity) -> str:
    return "\n".join([
        "🤖 BOT STARTED",
        "",
        f"Mode:     {mode}",
        f"Markets:  {', '.join(base_asset(s) for s in symbols)}",
        f"Balance:  {equity:,.2f} {mkt().currency}",
        "/status · /help",
    ])


def kill_switch(limit) -> str:
    return "\n".join([
        "🛑 KILL SWITCH",
        "",
        f"The daily loss reached {limit:.0%}.",
        "All positions were closed.",
        "No new trades until tomorrow (UTC).",
    ])


def review(trade_id, sym, side, lesson) -> str:
    return "\n".join([f"📝 TRADE REVIEW · #{trade_id}", f"{base_asset(sym)} {SIDE_WORD[side]}", "", lesson])
