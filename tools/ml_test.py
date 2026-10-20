"""Two questions about the AI filter (core/ml.py), owner request 2026-10-20:

  1. Meta-labeling: the score still picks long/short, a LightGBM model trained only
     on earlier candles says "take it" or "skip it". Must pass the same gates as
     every change: better walk-forward folds, vault not worse, the shared account
     makes money on the vault, and it beats random entries (p <= 0.20).
  2. Probability: when the model says 60%, do ~60% of those trades win? And does
     a bigger score already mean a better chance, without any model?
     (out-of-sample only: every number comes from a model that never saw that period)

    python tools/ml_test.py fx
    python tools/ml_test.py commodities
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

MIN_GAIN = 0.1


def run(market_name: str):
    import markets
    from core import improver, ml, portfolio, significance
    from core.context import activate, mkt
    from core.settings import DATA_ROOT
    from core.storage import Storage
    from core.strategy import compute, load_params

    activate(markets.get(market_name))
    frames = improver.load_frames(Storage(), mkt().symbols)
    p = {**mkt().default_params, **load_params()}
    ds = ml.Dataset(frames, p)
    purge = int(p["max_hold_bars"]) * ds.step

    out, oos = {"base": {}, "meta": {}}, []
    meta_sig = {s: np.zeros(len(d["ts"]), dtype=int) for s, d in ds.rows.items()}
    for name, a, b in ds.windows():
        model, thr = ds._train(a - purge, "meta")
        sig = {}
        for sym, d in ds.rows.items():
            win = (d["ts"] >= a) & (d["ts"] < b)
            s = np.zeros(len(d["ts"]), dtype=int)
            if model is not None:
                take = win & (d["primary"] != 0)
                prob = np.full(len(d["ts"]), np.nan)
                prob[take] = model.predict(d["X"][take])
                if thr < 1.0:
                    s[take] = np.where(prob[take] >= thr, d["primary"][take], 0)
                r = np.where(d["primary"] == 1, d["r_long"], d["r_short"])
                f = compute(d["df"], p)
                for i in np.nonzero(take & np.isfinite(r))[0]:
                    oos.append({"window": name, "symbol": sym, "prob": float(prob[i]), "r": float(r[i]),
                                "score": abs(float(f["score"].iloc[i]))})
            sig[sym] = s
            meta_sig[sym][win] = s[win]
        out["meta"][name] = {**ds.evaluate(sig, a, b), "threshold": thr}
        out["base"][name] = ds.evaluate(ds.signals("base", a, b)[0], a, b)

    def summary(res):
        folds = [v for k, v in res.items() if k != "vault"]
        return {"fold_score_mean": round(float(np.mean([f["score"] for f in folds])), 3),
                "positive_folds": int(sum(f["return"] > 0 for f in folds)),
                "vault": res["vault"]}

    report = {"base": summary(out["base"]), "meta": summary(out["meta"]), "windows": out}
    gain = report["meta"]["fold_score_mean"] - report["base"]["fold_score_mean"]
    report["wf_gain"] = round(gain, 3)
    if gain < MIN_GAIN:
        verdict = "❌ walk-forward сайжраагүй"
    elif report["meta"]["vault"]["score"] < report["base"]["vault"]["score"]:
        verdict = "❌ vault муудсан"
    else:
        feats = {}
        for sym, df in frames.items():
            f = compute(df, p)
            f["signal"] = meta_sig[sym]
            feats[sym] = f
        cut = max(int(df.index[int(len(df) * 0.85)]) for df in frames.values())
        report["portfolio"] = portfolio.run(frames, p, start_ts=cut, feats=feats)
        report["significance"] = significance.random_benchmark(frames, p, n=100, feats_by_symbol=feats)
        ok = report["portfolio"]["return"] > 0 and report["significance"]["p_value"] <= 0.20
        verdict = "✅ нэмэх нь зүйтэй" if ok else "❌ нэг данс / санамсаргүйтэй харьцуулалтад тэнцээгүй"
    report["verdict"] = verdict

    # ---- 2. probability: does it mean what it says? ----
    df = pd.DataFrame(oos)
    cal = {}
    if len(df) > 50:
        df["win"] = (df["r"] > 0).astype(float)
        base_rate = float(df["win"].mean())
        brier_model = float(((df["prob"] - df["win"]) ** 2).mean())
        brier_flat = float(((base_rate - df["win"]) ** 2).mean())
        bins = []
        for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)):
            g = df[(df["prob"] >= lo) & (df["prob"] < hi)]
            if len(g):
                bins.append({"prob": f"{lo:.0%}-{min(hi, 1):.0%}", "n": len(g), "said": round(float(g["prob"].mean()), 3),
                             "won": round(float(g["win"].mean()), 3), "avg_r": round(float(g["r"].mean()), 3)})
        score_bins = []
        qs = df["score"].quantile([0, .25, .5, .75, 1]).to_numpy()
        for k in range(4):
            g = df[(df["score"] >= qs[k]) & (df["score"] <= qs[k + 1] if k == 3 else df["score"] < qs[k + 1])]
            if len(g):
                score_bins.append({"score": f"{qs[k]:.2f}-{qs[k + 1]:.2f}", "n": len(g),
                                   "won": round(float(g["win"].mean()), 3), "avg_r": round(float(g["r"].mean()), 3)})
        cal = {"signals": len(df), "base_win_rate": round(base_rate, 3), "brier_model": round(brier_model, 4),
               "brier_no_model": round(brier_flat, 4), "skill": round(1 - brier_model / brier_flat, 3),
               "by_prob": bins, "by_score": score_bins}
    report["probability"] = cal

    path = DATA_ROOT / market_name / "ml_test.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    b, m = report["base"], report["meta"]
    print(f"== {mkt().label}")
    print(f"Одоогийнх:   folds {b['fold_score_mean']:+.2f} ({b['positive_folds']}/3 ашигтай), "
          f"vault {b['vault']['score']:+.2f}, {b['vault']['trades']} арилжаа, {b['vault']['return']:+.1%}")
    print(f"AI шүүлтүүр: folds {m['fold_score_mean']:+.2f} ({m['positive_folds']}/3 ашигтай), "
          f"vault {m['vault']['score']:+.2f}, {m['vault']['trades']} арилжаа, {m['vault']['return']:+.1%}")
    if "significance" in report:
        print(f"Нэг данс vault {report['portfolio']['return']:+.1%}, санамсаргүйтэй p={report['significance']['p_value']:.2f}")
    print("->", verdict)
    if cal:
        print(f"\nМагадлал: {cal['signals']} дохио (бүгд загвар хараагүй үеийнх), дунджаар {cal['base_win_rate']:.0%} нь ашигтай."
              f" Загварын ур чадвар {cal['skill']:+.1%} (0 = юу ч мэдэхгүй, + = сайн)")
        for x in cal["by_prob"]:
            print(f"  AI {x['prob']:>8}: {x['n']:5} дохио, хэлсэн {x['said']:.0%}, бодит {x['won']:.0%}, дундаж {x['avg_r']:+.2f}R")
        for x in cal["by_score"]:
            print(f"  оноо {x['score']:>10}: {x['n']:5} дохио, бодит {x['won']:.0%}, дундаж {x['avg_r']:+.2f}R")
    print("->", path, "\n")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fx", "commodities"]:
        run(name)
