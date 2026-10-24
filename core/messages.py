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
    from .backtest import entry_wait_candles
    wait_h = entry_wait_candles() * mkt().tf_ms() / 3_600_000   # counted from when it is placed
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


def improve_report(rep: dict, min_shadow_days: int) -> str:
    """The daily self-improvement run in plain words: what was tried, what passed,
    what changed (usually nothing: new settings must first win on new candles)."""
    cur = rep.get("current") or {}
    wf, v = cur.get("walk_forward") or {}, cur.get("vault") or {}
    best = rep.get("best_walk_forward") or {}
    width = {0.05: "narrow (things go well)", 0.10: "normal", 0.20: "wide (results are weak)"}
    lines = [
        f"🧠 DAILY SELF-IMPROVEMENT · {mkt().label}",
        "",
        f"Tried {rep.get('candidates', 0):,} new settings on the past "
        f"({width.get(rep.get('search_width'), 'normal')} search, dopamine {rep.get('dopamine', 0):+.2f})",
        "",
        "📏 Live settings on the past:",
        f"• periods with profit: {wf.get('positive_folds', 0)}/4, {wf.get('trades', 0)} trades",
        f"• held-back test (vault): {v.get('return', 0):+.1%}, {v.get('trades', 0)} trades, "
        f"avg {v.get('avg_r', 0):+.2f}R",
    ]
    if (rep.get("health") or {}).get("weak"):
        lines.append("⚠️ The live settings no longer pass their own tests -> risk per trade is HALVED "
                     "until they pass again or a shadow takes over")
    if best:
        lines.append(f"🥇 Best new one: {best.get('positive_folds', 0)}/4 profit periods, "
                     f"avg {best.get('avg_r', 0):+.2f}R")
    checked = rep.get("checked") or []
    if checked:
        why = {}
        for c in checked:
            if c.get("rejected"):
                why[c["rejected"]] = why.get(c["rejected"], 0) + 1
        en = {"vault оноо одоогийнхоос дээр биш": "not better than live on the held-back test",
              "vault дээр арилжаа цөөн": "too few trades to trust",
              "vault алдагдалтай / уналт их": "lost money or fell too much on the held-back test",
              "нэг данс эсвэл санамсаргүйтэй харьцуулалтад тэнцээгүй": "no better than random entries"}
        lines += ["", f"🔍 Checked the top {len(checked)} closely:"]
        lines += [f"• ✗ {n} × {en.get(k, k)}" for k, n in why.items()]
        passed = len(checked) - sum(why.values())
        lines.append(f"• ✓ {passed} passed every test" if passed else "• nothing passed every test")
    for s in rep.get("new_shadows", []):
        lines.append(f"🧪 New shadow {s['id']} (held-back test {s['vault'].get('return', 0):+.1%}, "
                     f"{s['vault'].get('trades', 0)} trades) starts trading on paper next to live")
    if rep.get("shadows_full"):
        lines.append(f"⏳ {rep['shadows_full']} more passed, but all shadow places are taken by tests still running")
    for les in rep.get("lessons") or []:
        for b in les:
            lines.append(f"📚 Lesson: stop entering when '{b['tag']}' ({b['trades']} past trades, "
                         f"avg {b['avg_r']:+.2f}R)")
    if rep.get("edges_refreshed"):
        lines.append("🔄 Chart-pattern scores updated")
    shadows = rep.get("shadow_review") or []
    if shadows:
        lines += ["", "👥 Shadows (paper, new candles only):"]
        for s in shadows:
            sh = s.get("shadow") or {}
            lines.append(f"• {s['id']} · {s.get('days', 0):.1f} days · {sh.get('trades', 0)} trades · "
                         f"{sh.get('return', 0):+.1%}")
    lines.append("")
    if rep.get("shadow_promoted"):
        lines.append(f"✅ CHANGED TODAY: shadow {rep['shadow_promoted']} beat live on new candles and is now live")
    else:
        lines.append(f"⏸ Live settings NOT changed today. A shadow must beat them on at least "
                     f"{min_shadow_days} days of new candles first.")
    return "\n".join(lines)
