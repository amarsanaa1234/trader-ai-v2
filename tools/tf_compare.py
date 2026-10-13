"""Which candle size works better for a market? Runs the same search (same seed,
same number of candidates, mistake memory, walk-forward ranking, vault check) on
each timeframe and prints/saves the comparison. Live settings are not touched.

    python tools/tf_compare.py fx 1h 4h [candidates]
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def run(market_name: str, tf: str, n: int) -> dict:
    os.environ[f"{market_name.upper()}_TIMEFRAME"] = tf      # read by Market.timeframe (and worker processes)
    import markets
    from core import improver, mistakes, validation
    from core.context import activate, mkt
    from core.storage import Storage
    from core.strategy import default_params

    activate(markets.get(market_name))
    st = Storage()
    frames = improver.load_frames(st, mkt().symbols)
    rng = random.Random(7)
    base = default_params()
    cands = [improver.mutate(base, 0.15, rng) for _ in range(int(n * 0.6))]
    cands += [improver.random_params(rng, base) for _ in range(n - len(cands))]
    results = improver.walk_forward_many(frames, cands)
    scored = sorted(zip(results, cands), key=lambda x: -x[0]["score"])
    lessons = []
    for wf, c in list(scored[:3]):
        better, learned = mistakes.learn(frames, c, wf)
        if learned:
            scored.append((validation.walk_forward(frames, better), better))
            lessons.append(learned)
    scored.sort(key=lambda x: -x[0]["score"])
    top = []
    for wf, c in scored[:5]:
        top.append({"walk_forward": wf, "vault": validation.vault(frames, c), "avoid": c.get("avoid", []),
                    "params": {k: c[k] for k in ("entry_threshold", "stop_atr", "tp_r", "max_hold_bars")}})
    return {"timeframe": tf, "bars": {s: len(f) for s, f in frames.items()},
            "default": {"walk_forward": validation.walk_forward(frames, base), "vault": validation.vault(frames, base)},
            "top": top, "lessons": lessons}


if __name__ == "__main__":
    market, *tfs = sys.argv[1:]
    n = int(tfs.pop()) if tfs and tfs[-1].isdigit() else 600
    out = {}
    for tf in tfs:
        out[tf] = run(market, tf, n)
        best = out[tf]["top"][0]
        print(f"{tf}: walk-forward {best['walk_forward']['score']:+.2f} "
              f"({best['walk_forward']['positive_folds']}/4 ашигтай, {best['walk_forward']['trades']} арилжаа), "
              f"vault {best['vault']['return']:+.2%} ({best['vault']['trades']} арилжаа, оноо {best['vault']['score']:+.2f})")
    from core.settings import DATA_ROOT
    path = DATA_ROOT / market / "tf_compare.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("->", path)
