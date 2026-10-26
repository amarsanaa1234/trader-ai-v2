"""Crypto-only CLI commands: python main.py crypto <command>."""
def carry_report(st):
    from . import carry
    print("Одоогийн funding (жилийн, сүүлийн 3 төлбөрийн дундаж):")
    for c in carry.scan():
        print(f"  {c['symbol']:16} {c['apr']:+.1%}")
    print("2 жилийн backtest:")
    for r in carry.backtest():
        print(f"  {r['symbol']:16} зах зээлд {r['time_in_market']:.0%} хугацаа, "
              f"хөрөнгөнд жилд {r['return_per_year_on_capital']:+.2%}")


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
    "liquidations": (liquidations_report, "сүүлийн 24 цагийн liquidation-ий дүн"),
    "liqrecord": (liqrecord, "liquidation цуглуулагч (тусдаа service)"),
}
