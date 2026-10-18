"""Try one exit setting at several values with everything else unchanged.

    python tools/exit_test.py fx be_at_r 0 0.5 0.75 1 1.5 2 2.5

The value is chosen on walk-forward (history only); the vault, the shared-account
vault and the random-entry benchmark are then shown for the chosen value and the
current one, so you can see it isn't just luck on the newest data.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def main(market_name, key, values):
    import markets
    from core import improver, portfolio, significance, validation
    from core.context import activate, mkt
    from core.storage import Storage
    from core.strategy import load_params

    activate(markets.get(market_name))
    frames = improver.load_frames(Storage(), mkt().symbols)
    base = load_params()
    rows = []
    for v in values:
        p = {**base, key: v}
        wf = validation.walk_forward(frames, p)
        rows.append({"value": v, "walk_forward": wf})
        print(f"{key}={v:<5} walk-forward {wf['score']:+.3f} ({wf['positive_folds']}/4, {wf['trades']} арилжаа, "
              f"дундаж {wf['avg_r']:+.3f}R, өгөөж {wf['fold_returns']})")
    best = max(rows, key=lambda r: r["walk_forward"]["score"])
    out = {"current": base[key], "rows": rows, "best": best["value"], "check": {}}
    for label, v in (("одоогийн", base[key]), ("шилдэг", best["value"])):
        p = {**base, key: v}
        out["check"][label] = {"value": v, "vault": validation.vault(frames, p), "portfolio": portfolio.vault(frames, p),
                               "significance": significance.random_benchmark(frames, p, n=100)}
        c = out["check"][label]
        print(f"{label} ({key}={v}): vault {c['vault']['return']:+.2%} ({c['vault']['trades']} арилжаа, "
              f"оноо {c['vault']['score']:+.2f}), нэг данс {c['portfolio']['return']:+.2%} уналт "
              f"{c['portfolio']['max_dd']:.1%}, санамсаргүйтэй p={c['significance']['p_value']}")
    from core.settings import DATA_ROOT
    (DATA_ROOT / market_name / f"exit_test_{key}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]])
