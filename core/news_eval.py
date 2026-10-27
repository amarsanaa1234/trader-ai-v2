"""Did the AI's news scores point the right way? (owner, 2026-10-19: decide after a
month whether news gets a say again.)

For every scored headline (one asset, not "MARKET", score not 0) the price move
after the score became known is measured 4h and 24h ahead, in ATRs, signed by the
score: positive = the AI was right. A real edge shows as more than half right with
a t-statistic above ~2, and strong scores doing better than weak ones.

Each score is used only from the moment it was known (sentiment.scored_ts; older
rows without it: 6 hours after publication, to be safe), so nothing peeks.

The ideas registered in advance (core/news_topics.HYPOTHESES) are checked
separately, on every headline known since news_topics.REGISTERED_FROM, so they
are judged only on data that did not exist when they were chosen.

The trader runs this weekly from NEWS_EVAL_FROM on and stores the result as state
"news_eval"; the web report page shows it.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import news_topics
from .context import mkt
from .indicators import atr
from .storage import Storage, now_ms

HOUR = 3_600_000
MIN_T, MIN_HIT = 2.0, 0.55


def _registered_ms() -> int:
    return int(datetime.strptime(news_topics.REGISTERED_FROM, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def measure(st: Storage, since_ms: int) -> pd.DataFrame:
    """One row per (headline score, hours ahead): h, score, signed move in ATR, known, topic."""
    rows = pd.read_sql_query(
        "SELECT ts, asset, score, scored_ts, model, news_id FROM sentiment "
        "WHERE asset != 'MARKET' AND score != 0 AND ts >= ?", st.db, params=(since_ms,))
    cols = ["h", "score", "signed", "known", "topic"]
    if rows.empty:
        return pd.DataFrame(columns=cols + ["model"])
    titles = pd.read_sql_query("SELECT id, ts, title FROM news WHERE ts >= ?", st.db, params=(since_ms,))
    by_id = titles.set_index("id")["title"]
    one_per_ts = titles.groupby("ts").filter(lambda g: len(g) == 1).set_index("ts")["title"]   # older rows: no news_id
    rows["title"] = rows["news_id"].map(by_id).fillna(rows["ts"].map(one_per_ts)).fillna("")
    rows["topic"] = rows["title"].map(news_topics.topic)
    rows["known"] = rows["scored_ts"].fillna(rows["ts"] + 6 * HOUR).astype("int64")
    by_asset = {s.split("/")[0]: s for s in mkt().symbols}      # crypto "BTC" -> "BTC/USDT:USDT"; FX/commodities as is
    out = []
    for asset, grp in rows.groupby("asset"):
        c = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf='1h' ORDER BY ts",
                              st.db, params=(by_asset.get(asset, asset),)).set_index("ts")
        if len(c) < 50:
            continue
        a = atr(c, 14).to_numpy()
        idx, o, cl = c.index.to_numpy(), c["open"].to_numpy(), c["close"].to_numpy()
        for known, score, tp in zip(grp["known"], grp["score"], grp["topic"]):
            i = int(np.searchsorted(idx, known, side="left"))     # first candle opening after it was known
            for h in (4, 24):
                j = i + h - 1
                if i < 1 or j >= len(idx) or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
                    continue
                out.append((h, score, np.sign(score) * (cl[j] - o[i]) / a[i - 1], known, tp))
    df = pd.DataFrame(out, columns=cols)
    df.attrs.update(models=sorted({str(x) for x in rows["model"].dropna()}), scored=len(rows))
    return df


def _stats(x: np.ndarray) -> dict:
    return {"n": len(x), "hit": round(float(np.mean(x > 0)), 3), "avg_atr": round(float(x.mean()), 3),
            "t": round(float(x.mean() / (x.std(ddof=1) + 1e-9) * np.sqrt(len(x))), 2)}


def hypotheses(st: Storage) -> list[dict]:
    """The ideas written down in advance, on headlines known since they were registered."""
    since = _registered_ms()
    df = measure(st, since - 2 * 86_400_000)      # a little earlier: a score can be known days after publication
    df = df[df["known"] >= since]
    out = []
    for hyp in news_topics.HYPOTHESES:
        x = df[(df["h"] == hyp["hours"]) & (df["topic"] == hyp["topic"])]["signed"].to_numpy()
        r = {**hyp, "since": news_topics.REGISTERED_FROM, "n": len(x)}
        if len(x) >= 3:
            r.update(_stats(x))
        if len(x) < news_topics.MIN_N:
            r["status"] = f"⏳ цуглуулж байна ({len(x)}/{news_topics.MIN_N})"
        elif r["t"] >= MIN_T and r["hit"] >= MIN_HIT:
            r["status"] = "✅ батлагдлаа → shadow тестээр жин өгч турших"
        elif r["t"] <= -MIN_T:
            r["status"] = "🔄 эсрэгээрээ ажиллаж байна"
        else:
            r["status"] = "❌ батлагдсангүй"
        out.append(r)
    return out


def evaluate(st: Storage, days: int = 30) -> dict:
    collected = {"headlines": st.db.execute("SELECT COUNT(*) c FROM news").fetchone()["c"],
                 "scores": st.db.execute("SELECT COUNT(*) c FROM sentiment").fetchone()["c"]}
    df = measure(st, now_ms() - days * 86_400_000)
    result = {"run_at": now_ms(), "days": days, "collected": collected,
              "scored": df.attrs.get("scored", 0), "rows": [],
              "models": df.attrs.get("models", []), "hypotheses": hypotheses(st)}
    if df.empty:
        result["verdict"] = "Үнэлгээтэй мэдээ алга"
        return result
    for h in (4, 24):
        g = df[df["h"] == h]
        for label, sel in (("бүгд", g), ("хүчтэй (|оноо| ≥ 0.5)", g[g["score"].abs() >= 0.5])):
            if len(sel) < 3:
                continue
            result["rows"].append({"hours": h, "subset": label, **_stats(sel["signed"].to_numpy())})
    result["verdict"] = verdict(result["rows"])
    return result


def verdict(rows: list) -> str:
    good = [r for r in rows if r["t"] >= MIN_T and r["hit"] >= MIN_HIT]
    bad = [r for r in rows if r["t"] <= -MIN_T and r["hit"] <= 1 - MIN_HIT]
    if good:
        return "✅ Мэдээ зөв чиглэлийг тогтвортой зааж байна → shadow тестээр жин өгч турших"
    if bad:
        return "🔄 Мэдээ тогтвортой ЭСРЭГ ажиллаж байна (гарсны дараа үнэ буцдаг) → эсрэг дохио болгож турших боломжтой"
    return "➖ Тодорхой давуу тал алга → мэдээ шийдвэрт оролцохгүй хэвээр"
