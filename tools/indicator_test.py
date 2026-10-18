"""Test candidate chart methods (core/ta_extra.py) before they may join the score.

  1. Alone: after each signal, did price move that way over the next day?
     (average move in ATRs, t-statistic; history part and vault part separately)
  2. Added to the current settings: best weight chosen on walk-forward, then the
     vault, the shared-account vault and the random-entry benchmark must agree.

    python tools/indicator_test.py fx
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

WEIGHTS = (-1.0, -0.5, 0.5, 1.0)
MIN_GAIN = 0.1


def alone(frames, name, horizon_days=1):
    from core.indicators import atr
    from core.validation import research_end
    parts = {"history": [], "vault": []}
    for df in frames.values():
        bars = max(1, int(round(horizon_days * 86_400_000 / np.min(np.diff(df.index.to_numpy()[:200])))))
        fwd = ((df["close"].shift(-bars) - df["close"]) / atr(df, 14)).to_numpy()
        sig = df[f"cand_{name}"].to_numpy()
        cut = research_end(len(df))
        for i in np.nonzero(sig)[0]:
            if np.isfinite(fwd[i]):
                parts["history" if i < cut else "vault"].append(sig[i] * fwd[i])
    out = {}
    for k, v in parts.items():
        a = np.asarray(v)
        out[k] = {"signals": len(a), "avg_move_atr": round(float(a.mean()), 3) if len(a) else 0.0,
                  "t": round(float(a.mean() / (a.std(ddof=1) + 1e-9) * np.sqrt(len(a))), 2) if len(a) > 2 else 0.0,
                  "right_way": round(float((a > 0).mean()), 3) if len(a) else 0.0}
    return out


def added(frames, name, base, base_wf, base_vault):
    from core import portfolio, significance, validation
    from core.context import mkt
    m = mkt()
    orig_sources, orig_names, orig_defaults = m.sources, m.source_names, m.default_params
    m.source_names = tuple(orig_names) + ("cand",)
    m.default_params = {**orig_defaults, "w_cand": 0.0}
    m.sources = lambda df, p: {**orig_sources(df, p), "cand": df[f"cand_{name}"]}
    try:
        tried = []
        for w in WEIGHTS:
            p = {**base, "w_cand": w}
            tried.append((validation.walk_forward(frames, p), w))
        wf, w = max(tried, key=lambda x: x[0]["score"])
        res = {"weight": w, "walk_forward": wf, "wf_gain": round(wf["score"] - base_wf["score"], 3)}
        if res["wf_gain"] < MIN_GAIN:
            res["verdict"] = "❌ walk-forward сайжраагүй"
            return res
        p = {**base, "w_cand": w}
        res["vault"] = validation.vault(frames, p)
        if res["vault"]["score"] < base_vault["score"]:
            res["verdict"] = "❌ vault муудсан"
            return res
        res["portfolio"] = portfolio.vault(frames, p)
        res["significance"] = significance.random_benchmark(frames, p, n=100)
        ok = res["portfolio"]["return"] > 0 and res["significance"]["p_value"] <= 0.20
        res["verdict"] = "✅ нэмэх нь зүйтэй" if ok else "❌ нэг данс / санамсаргүйтэй харьцуулалтад тэнцээгүй"
        return res
    finally:
        m.sources, m.source_names, m.default_params = orig_sources, orig_names, orig_defaults


def main(market_name):
    import markets
    from core import improver, validation
    from core.context import activate, mkt
    from core.storage import Storage
    from core.strategy import load_params
    from core.ta_extra import CANDIDATES

    activate(markets.get(market_name))
    frames = improver.load_frames(Storage(), mkt().symbols)
    for name, fn in CANDIDATES.items():
        for df in frames.values():
            df[f"cand_{name}"] = fn(df).to_numpy()
    base = load_params()
    base_wf, base_vault = validation.walk_forward(frames, base), validation.vault(frames, base)
    print(f"Одоогийн тохиргоо: walk-forward {base_wf['score']:+.2f}, vault оноо {base_vault['score']:+.2f}\n")
    report = {"base": {"walk_forward": base_wf, "vault": base_vault}}
    for name in CANDIDATES:
        a = alone(frames, name)
        r = added(frames, name, base, base_wf, base_vault)
        report[name] = {"alone": a, "added": r}
        h, v = a["history"], a["vault"]
        print(f"{name:12} ганцаараа: түүх {h['signals']} дохио, {h['avg_move_atr']:+.3f} ATR (t {h['t']:+.1f}), "
              f"vault {v['signals']} дохио, {v['avg_move_atr']:+.3f} ATR (t {v['t']:+.1f})")
        print(f"{'':12} нэмэхэд: жин {r['weight']:+.1f}, walk-forward {r['wf_gain']:+.3f} -> {r['verdict']}")
    from core.settings import DATA_ROOT
    path = DATA_ROOT / market_name / "indicator_test.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n->", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fx")
