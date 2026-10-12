"""Crypto-only CLI commands: python main.py crypto <command>."""
import logging
import time

from core.storage import Storage


def carry_report(st):
    from . import carry
    print("Одоогийн funding (жилийн, сүүлийн 3 төлбөрийн дундаж):")
    for c in carry.scan():
        print(f"  {c['symbol']:16} {c['apr']:+.1%}")
    print("2 жилийн backtest:")
    for r in carry.backtest():
        print(f"  {r['symbol']:16} зах зээлд {r['time_in_market']:.0%} хугацаа, "
              f"хөрөнгөнд жилд {r['return_per_year_on_capital']:+.2%}")


def s2run(st):
    """Strategy 2 shadow: paper-trades next to the main bot, never places orders."""
    from . import strategy2
    log = logging.getLogger("s2")
    while True:
        try:
            s = strategy2.step(Storage())
            if s:
                log.info("Стратеги 2 (shadow): %.2f $ (%+.2f%%)", s["equity"], (s["equity"] / s["start_equity"] - 1) * 100)
        except Exception:
            log.exception("Стратеги 2 алхам амжилтгүй")
        time.sleep(300)


def s2(st):
    from . import strategy2
    snap = strategy2.snapshot(st)
    if not snap:
        print("Стратеги 2 хараахан эхлээгүй байна")
        return
    side = {1: "long", -1: "short", 0: "—"}
    print(f"Стратеги 2 (shadow): {snap['equity']:,.2f} $ ({snap['return']:+.2%}), нийт байрлал {snap['exposure']:.0%}")
    for c, p in snap["coins"].items():
        print(f"  {c:4} momentum {side.get(p['momentum'], '—'):5}  premium {side.get(p['premium'], '—'):5}  "
              f"жин {p['weight']:+.1%}")


def liqrecord(st):
    from . import liquidations
    liquidations.run()


def liquidations_report(st):
    from . import liquidations
    rows = liquidations.summary(24)
    if not rows:
        print("Сүүлийн 24 цагт бичигдсэн liquidation алга (цуглуулагч ажиллаж байна уу?)")
        return
    print(f"Сүүлийн 24 цаг: long хаагдсан {sum(r[1] for r in rows):,.0f} $ · short хаагдсан "
          f"{sum(r[2] for r in rows):,.0f} $ · {sum(r[3] for r in rows)} удаа")
    for sym, lng, sht, n in rows[:15]:
        print(f"  {sym:14} long {lng:>14,.0f} $   short {sht:>14,.0f} $   ({n})")


COMMANDS = {
    "carry": (carry_report, "funding carry: одоогийн funding + 2 жилийн backtest"),
    "s2": (s2, "Стратеги 2 (shadow)-ын одоогийн байдал"),
    "s2run": (s2run, "Стратеги 2 shadow-ыг тасралтгүй ажиллуулах (захиалга тавихгүй)"),
    "liquidations": (liquidations_report, "сүүлийн 24 цагийн liquidation-ий дүн"),
    "liqrecord": (liqrecord, "liquidation цуглуулагч (тусдаа service)"),
}
