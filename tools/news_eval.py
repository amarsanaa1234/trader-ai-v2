"""Did the AI's news scores point the right way? (see core/news_eval.py)

    python tools/news_eval.py crypto [days_back]

The bots also run this by themselves every week from NEWS_EVAL_FROM (default
2026-11-19) and show it on the web report page.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def main(market_name: str, days: int):
    import markets
    from core import news_eval
    from core.context import activate
    from core.storage import Storage

    m = markets.get(market_name)
    activate(m)
    r = news_eval.evaluate(Storage(), days)
    print(f"{m.label}: сүүлийн {days} хоногийн {r['scored']} мэдээний үнэлгээ ({', '.join(r['models']) or '?'})\n")
    for x in r["rows"]:
        print(f"{x['hours']:>2} цагийн дараа, {x['subset']:22}: {x['n']:4} мэдээ, зөв чиглэл {x['hit']:.0%}, "
              f"дундаж {x['avg_atr']:+.2f} ATR, t {x['t']:+.1f}")
    print("\n" + r["verdict"])


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
