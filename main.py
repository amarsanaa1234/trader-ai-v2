"""Trader AI v2 — python main.py <зах зээл> <команд>

Зах зээл: crypto (дараа нь fx, metals, commodities, indices, bonds)

Бүх зах зээлийн командууд:
  sync        түүхэн өгөгдөл татах
  backtest    одоогийн тохиргоог 4 хугацаа + vault дээр шалгах
  library     стратегийн санг backtest хийж, жагсаалт харах
  research    интернэтээс шинэ стратеги хайх (AI хэрэгтэй)
  improve     өөрийгөө сайжруулах (walk-forward -> vault -> shadow)
  shadows     shadow test-үүдийн байдал
  news        мэдээ уншиж, AI-аар үнэлэх
  events      удахгүй болох чухал үйл явдлууд (CPI, FOMC...)
  check       брокерийн холболт шалгах (зөвхөн уншина)
  status      арилжааны дүн, допамин
  report      7 хоногийн тайлан
  dashboard   зөвхөн dashboard асаах
  telegram    Telegram ботоо холбох (chat ID-г автоматаар олно)
  resetpeak   30%-ийн drawdown зогсолтыг шалгаад дахин нээх
  run         ботыг тасралтгүй ажиллуулах (dashboard хамт асна)

Жишээ:  python main.py crypto backtest
        python main.py crypto help      (тухайн зах зээлийн нэмэлт командууд)
"""
import json
import logging
import sys

import markets
from core.context import activate, mkt


def setup_logging():
    # Mongolian text must not crash the bot when Windows' console/pipe isn't UTF-8
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    from logging.handlers import RotatingFileHandler
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    # log file never grows past ~5 x 5MB, however long the bot runs
    handlers = [logging.StreamHandler(sys.stdout),
                RotatingFileHandler(mkt().log_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for name in ("urllib3", "httpx", "anthropic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def frames(st, static=False):
    from core import library
    from core.features import add_static_features
    out = {s: mkt().load_frame(st, s) for s in mkt().symbols}
    out = {s: f for s, f in out.items() if not f.empty}
    if static:
        strats = library.approved()
        out = {s: add_static_features(f, strats) for s, f in out.items()}
    return out


def cmd_sync(st):
    from core import library
    mkt().sync(st)
    library.ensure_seeded()
    for s in mkt().symbols:
        n = st.db.execute("SELECT COUNT(*) c FROM candles WHERE symbol=? AND tf=?", (s, mkt().timeframe)).fetchone()["c"]
        print(f"{s}: {n} лаа хадгалагдсан")


def cmd_backtest(st):
    from core import backtest, validation
    from core.strategy import load_params
    p = load_params()
    fr = frames(st, static=True)
    print("Walk-forward (4 хугацаа):", json.dumps(validation.walk_forward(fr, p), ensure_ascii=False))
    print("Vault (сүүлийн 15%):     ", json.dumps(validation.vault(fr, p), ensure_ascii=False))
    for s, df in fr.items():
        cut = validation.research_end(len(df))
        print(f"{s} vault дэлгэрэнгүй: {backtest.run(df, p, start=cut).summary()}")


def cmd_library(st):
    from core import library
    library.ensure_seeded()
    library.review(frames(st), only_pending=False)
    for s in library.load():
        wf = s.get("stats", {}).get("walk_forward", {})
        v = s.get("stats", {}).get("vault", {})
        mark = {"approved": "✅", "rejected": "❌"}.get(s["status"], "⏳")
        print(f"{mark} {s['name'][:40]:40} ашигтай хугацаа {wf.get('positive_folds')}/4, "
              f"wf оноо {wf.get('score')}, vault өгөөж {v.get('return')}  [{s.get('source', '')[:40]}]")


def cmd_research(st):
    from core import library, research
    new = research.research(st)
    print(f"{len(new)} шинэ стратеги олдлоо:")
    for s in new:
        print(" -", s["name"], "|", s["source"])
    if new:
        for s in library.review(frames(st)):
            print(f"   {s['name']}: {s['status']}")


def cmd_improve(st):
    from core import improver
    rep = improver.improve(st, mkt().symbols, dopamine=st.get("dopamine", 0.0))
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


def cmd_shadows(st):
    from core import shadow
    rep = shadow.review(st, frames(st, static=True))
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))


def cmd_news(st):
    from core import news
    from core.settings import base_asset
    print("Татсан гарчиг:", news.fetch_headlines(st))
    print("AI үнэлгээ:", news.score_news(st))
    for s in mkt().symbols:
        print(f"{base_asset(s)} мэдээний сэтгэгдэл: {news.current_sentiment(st, base_asset(s)):+.2f}")


def cmd_events(st):
    from datetime import datetime

    from core import events
    events.refresh(st)
    for e in events.upcoming(st, hours=24 * 7):
        print(datetime.fromtimestamp(e["ts"] / 1000).strftime("%Y-%m-%d %H:%M"), e["title"])
    print("Одоо арилжаа хаалттай юу:", events.blackout(st) or "үгүй")


def cmd_check(st):
    b = mkt().make_broker(st)
    print("Зах зээл:", mkt().name, "| горим:", mkt().mode)
    print("Equity:", round(b.equity(), 2))
    print("Position:", b.positions())


def cmd_status(st):
    from core import reward
    from core.storage import now_ms
    mode = mkt().mode
    dop = st.get("dopamine", 0.0)
    started = st.get("paper_started")
    print(f"{mkt().label} · горим: {mode} | допамин {dop:+.2f} {reward.mood(dop)} | "
          f"{'⏸ ЗОГССОН' if st.get('paused') else 'ажиллаж байна'}")
    if started:
        print(f"Туршилтын хугацаа: {(now_ms() - started) / 86_400_000:.1f} хоног")
    print("Өнөөдөр:", st.get("day"), "| оргил equity:", st.get("peak_equity"))
    trades = st.closed_trades(mode, 20)
    if trades:
        tot = sum(t["pnl"] or 0 for t in trades)
        wins = sum(1 for t in trades if (t["pnl"] or 0) > 0)
        print(f"Сүүлийн {len(trades)} арилжаа: нийт {tot:+.2f}, ялалт {wins}/{len(trades)}")
        for t in trades[:10]:
            print(f"  {t['symbol']:16} {t['side']:5} {t['exit_reason']:18} {t['pnl']:+8.2f} "
                  f"{t['r_multiple']:+.2f}R шагнал {t['reward']:+.2f}  {t.get('review') or ''}")
    for t in st.open_trades(mode):
        print(f"  НЭЭЛТТЭЙ: {t['symbol']} {t['side']} @ {t['entry']:.2f} stop {t['stop']:.2f} tp {t['tp']:.2f}")
    for c in st.open_carries(mode):
        print(f"  CARRY: {c['symbol']} {c['qty']:.4f}, funding {c['funding'] or 0:+.2f}")


def cmd_report(st):
    from core import review
    print(review.weekly_report(st, mkt().mode))


def cmd_dashboard(st):
    from core import dashboard
    from core.settings import DASHBOARD_PORT
    print(f"http://127.0.0.1:{DASHBOARD_PORT}  (Ctrl+C зогсооно)")
    dashboard.serve(background=False)


def cmd_telegram(st):
    """Finds your chat id after you send /start to your bot, saves it to .env, sends a test message."""
    import re

    import requests
    from core.settings import ROOT, TELEGRAM_BOT_TOKEN
    if not TELEGRAM_BOT_TOKEN:
        print("Эхлээд .env дотор TELEGRAM_BOT_TOKEN=... гэж @BotFather-ийн өгсөн token-оо бичнэ үү.")
        return
    api = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    me = requests.get(f"{api}/getMe", timeout=15).json()
    if not me.get("ok"):
        print("Token буруу байна:", me.get("description"))
        return
    print(f"Бот: @{me['result']['username']}")
    updates = requests.get(f"{api}/getUpdates", timeout=15).json().get("result", [])
    chats = [u["message"]["chat"] for u in updates if u.get("message", {}).get("chat", {}).get("type") == "private"]
    if not chats:
        print(f"Telegram дээр @{me['result']['username']} ботоо нээгээд /start гэж бичээд энэ командыг дахин ажиллуулна уу.")
        return
    chat = chats[-1]
    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8") if env.exists() else ""
    line = f"TELEGRAM_CHAT_ID={chat['id']}"
    text = re.sub(r"(?m)^TELEGRAM_CHAT_ID=.*$", line, text) if "TELEGRAM_CHAT_ID=" in text else text + f"\n{line}\n"
    env.write_text(text, encoding="utf-8")
    requests.post(f"{api}/sendMessage", timeout=15, json={
        "chat_id": chat["id"], "text": "✅ Trader AI v2 холбогдлоо! /help гэж бичээд командуудыг харна уу."})
    print(f"Chat ID {chat['id']} ({chat.get('first_name', '')}) .env-д хадгалагдлаа.")


def cmd_resetpeak(st):
    eq = mkt().make_broker(st).equity()
    st.set("peak_equity", eq)
    print(f"Оргил equity = {eq:.2f}")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in markets.AVAILABLE:
        print(__doc__)
        return
    activate(markets.get(sys.argv[1]))
    setup_logging()
    cmd = sys.argv[2]
    if cmd == "run":
        from core.trader import Trader
        Trader().run()
        return
    extra = mkt().commands()
    fn = globals().get(f"cmd_{cmd}") or (extra[cmd][0] if cmd in extra else None)
    if not fn:
        print(__doc__)
        if extra:
            print(f"{mkt().label}-ийн нэмэлт командууд:")
            for name, (_, text) in extra.items():
                print(f"  {name:14}{text}")
        return
    from core.storage import Storage
    fn(Storage())


if __name__ == "__main__":
    main()
