"""Did the AI's news scores point the right way? Run after a month of collecting.

For every scored headline (one asset, not "MARKET", score not 0) the price move
after the score became known is measured 4h and 24h ahead, in ATRs, signed by the
score: positive = the AI was right. A real edge shows as more than half right
with a t-statistic above ~2, and stronger scores doing better than weak ones.

Only uses each score from the moment it was known (scored_ts; older rows without
it: 6 hours after publication, to be safe), so nothing peeks.

    python tools/news_eval.py crypto [days_back]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

HOUR = 3_600_000


def main(market_name: str, days: int):
    import markets
    from core.context import activate
    from core.indicators import atr
    from core.storage import Storage, now_ms

    m = markets.get(market_name)
    activate(m)
    st = Storage()
    rows = pd.read_sql_query(
        "SELECT ts, asset, score, scored_ts, model FROM sentiment WHERE asset != 'MARKET' AND score != 0 AND ts >= ?",
        st.db, params=(now_ms() - days * 86_400_000,))
    if rows.empty:
        print("Үнэлгээтэй мэдээ алга.")
        return
    rows["known"] = rows["scored_ts"].fillna(rows["ts"] + 6 * HOUR).astype("int64")
    by_asset = {s.split("/")[0]: s for s in m.symbols}      # crypto "BTC" -> "BTC/USDT:USDT"; FX/commodities as is
    out = []
    for asset, grp in rows.groupby("asset"):
        sym = by_asset.get(asset, asset)
        c = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf='1h' ORDER BY ts",
                              st.db, params=(sym,)).set_index("ts")
        if len(c) < 50:
            continue
        a = atr(c, 14).to_numpy()
        idx, o, cl = c.index.to_numpy(), c["open"].to_numpy(), c["close"].to_numpy()
        for _, r in grp.iterrows():
            i = int(np.searchsorted(idx, r["known"], side="left"))   # first candle that opens after it was known
            for h in (4, 24):
                j = i + h - 1
                if i < 1 or j >= len(idx) or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
                    continue
                move = (cl[j] - o[i]) / a[i - 1]
                out.append({"asset": asset, "h": h, "score": r["score"], "signed": np.sign(r["score"]) * move,
                            "model": r["model"] or "?"})
    if not out:
        print("Хэмжих хангалттай үнийн өгөгдөл алга.")
        return
    df = pd.DataFrame(out)
    print(f"{m.label}: сүүлийн {days} хоногийн {len(rows)} мэдээний үнэлгээ\n")
    for h, g in df.groupby("h"):
        for label, sel in (("бүгд", g), ("хүчтэй (|оноо|≥0.5)", g[g["score"].abs() >= 0.5])):
            if len(sel) < 3:
                continue
            x = sel["signed"].to_numpy()
            t = x.mean() / (x.std(ddof=1) + 1e-9) * np.sqrt(len(x))
            print(f"{h:>2} цагийн дараа, {label:22}: {len(x):4} мэдээ, зөв чиглэл {np.mean(x > 0):.0%}, "
                  f"дундаж {x.mean():+.2f} ATR, t {t:+.1f}")
    print("\nДүгнэлт: t > 2 ба зөв чиглэл > 55% бол мэдээ ашигтай байж магадгүй -> w_news-ийг shadow тестээр турших.")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
