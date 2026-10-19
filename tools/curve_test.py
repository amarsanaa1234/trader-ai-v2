"""Test the futures-curve pieces of the commodities market (markets/commodities/curve.py)
before they get a say, the same way as every new source:

  1. Alone: after a strong backwardation / contango reading, did price move that way
     over the next 5 days? (history and vault parts separately)
  2. Added to the current settings: best value on walk-forward, then the vault, the
     shared-account vault and the random-entry benchmark must agree.
     Pieces: w_curve, w_curve_chg, swap_block.

    python tools/curve_test.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

TRIES = {"w_curve": (-1.0, -0.5, 0.5, 1.0), "w_curve_chg": (-1.0, -0.5, 0.5, 1.0),
         "swap_block": (0.1, 0.2, 0.3, 0.5)}
MIN_GAIN = 0.1


def alone(frames, col="curve", horizon_days=5, cut_off=0.1):
    """Average forward move in ATRs, signed by the curve (only readings beyond +-cut_off a year)."""
    from core.indicators import atr
    from core.validation import research_end
    parts = {"history": [], "vault": []}
    for df in frames.values():
        bars = max(1, int(round(horizon_days * 86_400_000 / np.min(np.diff(df.index.to_numpy()[:200])))))
        fwd = ((df["close"].shift(-bars) - df["close"]) / atr(df, 14)).to_numpy()
        c = df["carry"].to_numpy() if col == "curve" else df["carry"].diff(20 * bars // horizon_days).to_numpy()
        sig = np.where(c > cut_off, 1, np.where(c < -cut_off, -1, 0))
        cut = research_end(len(df))
        for i in np.nonzero(sig)[0][::bars]:            # one reading per horizon, not overlapping
            if np.isfinite(fwd[i]):
                parts["history" if i < cut else "vault"].append(sig[i] * fwd[i])
    out = {}
    for k, v in parts.items():
        a = np.asarray(v)
        out[k] = {"signals": len(a), "avg_move_atr": round(float(a.mean()), 3) if len(a) else 0.0,
                  "t": round(float(a.mean() / (a.std(ddof=1) + 1e-9) * np.sqrt(len(a))), 2) if len(a) > 2 else 0.0,
                  "right_way": round(float((a > 0).mean()), 3) if len(a) else 0.0}
    return out


def added(frames, key, base, base_wf, base_vault):
    from core import portfolio, significance, validation
    tried = [(validation.walk_forward(frames, {**base, key: v}), v) for v in TRIES[key]]
    wf, v = max(tried, key=lambda x: x[0]["score"])
    res = {"value": v, "walk_forward": wf, "wf_gain": round(wf["score"] - base_wf["score"], 3),
           "all": {str(val): round(w["score"], 3) for w, val in tried}}
    if res["wf_gain"] < MIN_GAIN:
        res["verdict"] = "❌ walk-forward сайжраагүй"
        return res
    p = {**base, key: v}
    res["vault"] = validation.vault(frames, p)
    if res["vault"]["score"] < base_vault["score"]:
        res["verdict"] = "❌ vault муудсан"
        return res
    res["portfolio"] = portfolio.vault(frames, p)
    res["significance"] = significance.random_benchmark(frames, p, n=100)
    ok = res["portfolio"]["return"] > 0 and res["significance"]["p_value"] <= 0.20
    res["verdict"] = "✅ нэмэх нь зүйтэй" if ok else "❌ нэг данс / санамсаргүйтэй харьцуулалтад тэнцээгүй"
    return res


def main():
    import markets
    from core import improver, validation
    from core.context import activate, mkt
    from core.settings import DATA_ROOT
    from core.storage import Storage
    from core.strategy import load_params

    activate(markets.get("commodities"))
    frames = improver.load_frames(Storage(), mkt().symbols)
    have = {s: int(np.isfinite(df["carry"]).sum()) for s, df in frames.items()}
    print("Муруйн утгатай лаа:", have, "\n")
    base = {**mkt().default_params, **load_params()}
    base_wf, base_vault = validation.walk_forward(frames, base), validation.vault(frames, base)
    print(f"Одоогийн тохиргоо (шөнийн хүү тухайн үеийн муруйгаар): walk-forward {base_wf['score']:+.2f}, "
          f"vault {base_vault['score']:+.2f} ({base_vault.get('return', 0):+.1%})\n")
    report = {"base": {"walk_forward": base_wf, "vault": base_vault}, "candles_with_curve": have}
    for col in ("curve", "curve_chg"):
        a = alone(frames, col)
        report[f"alone_{col}"] = a
        h, v = a["history"], a["vault"]
        print(f"{col:10} ганцаараа (5 хоногийн дараа): түүх {h['signals']} дохио {h['avg_move_atr']:+.3f} ATR "
              f"(t {h['t']:+.1f}, зөв {h['right_way']:.0%}), vault {v['signals']} дохио {v['avg_move_atr']:+.3f} ATR "
              f"(t {v['t']:+.1f}, зөв {v['right_way']:.0%})")
    print()
    winners = {}
    for key in TRIES:
        r = added(frames, key, base, base_wf, base_vault)
        report[key] = r
        print(f"{key:12} хамгийн сайн {r['value']:+.2f}: walk-forward {r['wf_gain']:+.3f} {r['all']} -> {r['verdict']}")
        if r["verdict"].startswith("✅"):
            winners[key] = r["value"]
    if len(winners) > 1:
        from core import portfolio, significance
        p = {**base, **winners}
        both = {"params": winners, "walk_forward": validation.walk_forward(frames, p),
                "vault": validation.vault(frames, p), "portfolio": portfolio.vault(frames, p),
                "significance": significance.random_benchmark(frames, p, n=100)}
        report["together"] = both
        print(f"\nХамтдаа {winners}: walk-forward {both['walk_forward']['score']:+.2f}, "
              f"vault {both['vault']['score']:+.2f}, p={both['significance']['p_value']:.2f}")
    report["winners"] = winners
    path = DATA_ROOT / "commodities" / "curve_test.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n->", path)


if __name__ == "__main__":
    main()
