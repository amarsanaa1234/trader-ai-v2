"""Trader AI v2

Бүх зах зээлд нийтлэг:
  python main.py web                     вэб (нэвтрэлт, цэс, хүн бүрийн хуудас), http://127.0.0.1:8765
  python main.py users                   хэрэглэгчид (.env USER<n>_...), тэдний зах зээл, нууц үг
  python main.py user password <нэр>     вэбийн нууц үг тохируулах
  python main.py brain2                  2 дахь тархи: 1 цагийн лаан дээрх paper бот (зөвхөн харьцуулалт)

Зах зээл тус бүр: python main.py <зах зээл> <команд>

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
  status      хүн бүрийн дүн, допамин
  report      7 хоногийн тайлан
  migrate     v1-ийн нэг хэрэглэгчийн өгөгдлийг гол админд шилжүүлэх (нэг удаа)
  telegram    Telegram ботоо холбох (chat ID-г автоматаар олно)
  resetpeak   30%-ийн drawdown зогсолтыг шалгаад дахин нээх
  run         ботыг тасралтгүй ажиллуулах (бүх хэрэглэгчийн данс)

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
    from core import backtest, portfolio, significance, validation
    from core.strategy import load_params
    p = load_params()
    fr = frames(st, static=True)
    print("Walk-forward (4 хугацаа):", json.dumps(validation.walk_forward(fr, p), ensure_ascii=False))
    print("Vault (сүүлийн 15%):     ", json.dumps(validation.vault(fr, p), ensure_ascii=False))
    print("Нэг данс, vault:         ", json.dumps(portfolio.vault(fr, p), ensure_ascii=False))
    print("Санамсаргүйтэй харьцуулалт:", json.dumps(significance.random_benchmark(fr, p, n=100), ensure_ascii=False))
    for s, df in fr.items():
        cut = validation.research_end(len(df))
        print(f"{s} vault дэлгэрэнгүй: {backtest.run(df, p, start=cut, costs=mkt().costs_for(s)).summary()}")


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


def accounts(st):
    from core import users
    from core.account import Account
    out = []
    for u in mkt().traders(users.load_users(markets.AVAILABLE)):
        try:
            out.append(Account(u, st))
        except Exception as e:
            print(f"  {u.name}: данс холбогдсонгүй: {e}")
    return out


def cmd_check(st):
    print("Зах зээл:", mkt().name, "| горим:", mkt().mode)
    for acc in accounts(st):
        print(f"👤 {acc.name}: equity {acc.broker.equity():,.2f} | position {acc.broker.positions()}")


def cmd_status(st):
    from core import reward, users
    from core.storage import now_ms
    mode = mkt().mode
    started = st.get("paper_started")
    print(f"{mkt().label} · горим: {mode}")
    if started:
        print(f"Туршилтын хугацаа: {(now_ms() - started) / 86_400_000:.1f} хоног")
    for u in mkt().traders(users.load_users(markets.AVAILABLE)):
        g = lambda k, d=None: st.get(f"{u.name}:{k}", d)  # noqa: E731
        dop = g("dopamine", 0.0)
        print(f"\n👤 {u.name}: допамин {dop:+.2f} {reward.mood(dop)} | {'⏸ ЗОГССОН' if g('paused') else 'ажиллаж байна'}")
        print("   Өнөөдөр:", g("day"), "| оргил equity:", g("peak_equity"))
        _print_trades(st, mode, u.name)


def _print_trades(st, mode, user):
    trades = st.closed_trades(mode, 20, user)
    if trades:
        tot = sum(t["pnl"] or 0 for t in trades)
        wins = sum(1 for t in trades if (t["pnl"] or 0) > 0)
        print(f"Сүүлийн {len(trades)} арилжаа: нийт {tot:+.2f}, ялалт {wins}/{len(trades)}")
        for t in trades[:10]:
            print(f"  {t['symbol']:16} {t['side']:5} {t['exit_reason']:18} {t['pnl']:+8.2f} "
                  f"{t['r_multiple']:+.2f}R шагнал {t['reward']:+.2f}  {t.get('review') or ''}")
    for t in st.open_trades(mode, user):
        print(f"  НЭЭЛТТЭЙ: {t['symbol']} {t['side']} @ {t['entry']:.2f} stop {t['stop']:.2f} tp {t['tp']:.2f}")
    for c in st.open_carries(mode, user):
        print(f"  CARRY: {c['symbol']} {c['qty']:.4f}, funding {c['funding'] or 0:+.2f}")


def cmd_report(st):
    from core import review
    print(review.weekly_report(st, mkt().mode))


def cmd_migrate(st):
    """v1 kept one person's data without an owner: hand it to the main admin (or argv[3])."""
    from core import users
    who = sys.argv[3] if len(sys.argv) > 3 else users.primary(users.load_users(markets.AVAILABLE)).name
    print(f"{mkt().label}: {st.migrate_to_user(who)} мөр/түлхүүр → {who}")


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
    # multi-user .env: the chat belongs to the main admin (USER<n>_TELEGRAM_CHAT_ID)
    from core import users
    lead = users.primary(users.load_users(markets.AVAILABLE))
    key = "TELEGRAM_CHAT_ID" if lead is None or lead.legacy else f"USER{lead.id}_TELEGRAM_CHAT_ID"
    line = f"{key}={chat['id']}"
    if re.search(rf"(?m)^{key}=", text):
        text = re.sub(rf"(?m)^{key}=.*$", line, text)
    else:
        text = text.rstrip("\n") + f"\n{line}\n"
    env.write_text(text, encoding="utf-8")
    requests.post(f"{api}/sendMessage", timeout=15, json={
        "chat_id": chat["id"], "text": "✅ Trader AI v2 холбогдлоо! /help гэж бичээд командуудыг харна уу."})
    print(f"Chat ID {chat['id']} ({chat.get('first_name', '')}) .env-д хадгалагдлаа.")


def cmd_resetpeak(st):
    for acc in accounts(st):
        eq = acc.broker.equity()
        acc.set("peak_equity", eq)
        print(f"{acc.name}: оргил equity = {eq:.2f}")


def global_command(args) -> bool:
    """Commands that span every market. Returns False if args aren't one."""
    from core import auth, users
    if args[:1] == ["web"]:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        from core import web
        web.serve()
        return True
    if args[:1] == ["brain2"]:
        # brain 2: the paper bot on 1h candles for every market, comparison only (core/brain2.py)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        from core import brain2
        brain2.run_forever()
        return True
    if args[:1] == ["users"]:
        everyone = users.load_users(markets.AVAILABLE)
        for u in everyone:
            active = []
            for name in markets.AVAILABLE:
                activate(markets.get(name))
                if name in u.markets:
                    active.append(name + ("" if u.name in {t.name for t in mkt().traders(everyone)} else " (түлхүүр алга)"))
            print(f"{u.id:>2} {u.name:14} {u.role:6} нууц үг {'✅' if auth.has_password(u.name) else '❌'}  "
                  f"Telegram {'✅' if u.telegram_chat_id else '❌'}  зах зээл: {', '.join(active) or '—'}")
        return True
    if args[:2] == ["user", "password"] and len(args) == 3:
        import getpass
        name = args[2].lower()
        if not users.by_name(users.load_users(markets.AVAILABLE), name):
            print(f"{name}: .env дотор ийм хэрэглэгч алга (USER<n>_NAME={name})")
            return True
        p1 = getpass.getpass(f"{name}-ийн шинэ нууц үг (≥{auth.MIN_PASSWORD}): ")
        if p1 != getpass.getpass("Дахин: "):
            print("Таарахгүй байна.")
            return True
        auth.set_password(name, p1)
        print("✅ Хадгаллаа (hash хэлбэрээр).")
        return True
    return False


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if global_command(sys.argv[1:]):
        return
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
