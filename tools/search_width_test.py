"""Does a wider self-improvement search find better crypto settings?

The daily improver picks its search from dopamine: live profit -> narrow (600
candidates, +-5% steps), losses -> wide (1600, +-20%). Crypto's demo profit came
from v1 while its backtest is weak, so it has been searching narrowly. This runs
the same candidate generation and the same gates as core/improver.py for each
width, without saving anything, and prints how far each one got.

    python tools/search_width_test.py <live strategy_params.json> [market]
"""
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[2] if len(sys.argv) > 2 else "crypto"))

from core import improver, patterns, validation  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params, default_params  # noqa: E402

WIDTHS = [("narrow (dopamine > 0.3)", 0.05, 600), ("normal", 0.10, 1000), ("wide (dopamine < -0.2)", 0.20, 1600)]
SEEDS = [1, 2]
GATES_FOR = 3      # portfolio + random-entry test only for the first few that pass the vault (slow)


def run_width(frames, current, cur_wf, cur_vault, scale, n, seed):
    rng = random.Random(seed)
    cands = [improver.mutate(current, scale, rng) for _ in range(int(n * 0.8))]
    cands += [improver.random_params(rng, current) for _ in range(n - len(cands))]
    for c in cands:
        c["pattern_edges"] = current["pattern_edges"]
    t0 = time.time()
    results = improver.walk_forward_many(frames, cands)
    scored = sorted(zip(results, cands), key=lambda x: -x[0]["score"])
    beat_wf = [(wf, c) for wf, c in scored
               if wf["score"] > cur_wf["score"] and wf["positive_folds"] >= improver.MIN_POSITIVE_FOLDS]
    out = {"candidates": n, "seconds": round(time.time() - t0), "best_wf": scored[0][0]["score"],
           "best_wf_folds": scored[0][0]["positive_folds"], "beat_live_wf": len(beat_wf),
           "top10": [], "passed_vault": 0, "passed_all": 0}
    gated = 0
    for wf, cand in beat_wf[:10]:
        v = validation.vault(frames, cand)
        row = {"wf": wf["score"], "folds": wf["positive_folds"], "vault_ret": v["return"], "vault_trades": v["trades"],
               "vault_score": v["score"], "vault_dd": v["max_dd"]}
        ok = (v["score"] > cur_vault["score"] + improver.PROMOTE_MARGIN and v["trades"] >= improver.min_vault_trades()
              and v["return"] > 0 and v["max_dd"] < improver.MAX_VAULT_DD)
        row["vault_ok"] = ok
        if ok:
            out["passed_vault"] += 1
            if gated < GATES_FOR:
                gated += 1
                all_ok, extra = improver.extra_gates(frames, cand)
                row["portfolio_ret"] = (extra.get("portfolio") or {}).get("return")
                row["p_value"] = (extra.get("significance") or {}).get("p_value")
                row["all_ok"] = all_ok
                out["passed_all"] += int(all_ok)
        out["top10"].append(row)
    return out


def main():
    st = Storage()
    frames = improver.load_frames(st, mkt().symbols)
    current = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # missing newer keys get their defaults, like load_params() does
    current = clamp_params({**default_params(), **current.get("params", current)})
    current["pattern_edges"] = patterns.measure_edges(frames, 1 - validation.VAULT_FRACTION)
    cur_wf, cur_vault = validation.walk_forward(frames, current), validation.vault(frames, current)
    print(f"LIVE {mkt().name}: walk-forward {cur_wf['score']} ({cur_wf['positive_folds']}/4 folds, "
          f"{cur_wf['trades']} trades) | vault {cur_vault['return']:+.1%} {cur_vault['trades']} trades "
          f"score {cur_vault['score']}")
    report = {"live": {"walk_forward": cur_wf, "vault": cur_vault}, "runs": []}
    for label, scale, n in WIDTHS:
        for seed in SEEDS:
            r = run_width(frames, current, cur_wf, cur_vault, scale, n, seed)
            r.update(label=label, scale=scale, seed=seed)
            report["runs"].append(r)
            best_v = max((x["vault_ret"] for x in r["top10"]), default=None)
            print(f"{label:24} seed {seed}: best wf {r['best_wf']} ({r['best_wf_folds']}/4) | "
                  f"{r['beat_live_wf']} beat live on walk-forward | top10 best vault "
                  f"{'-' if best_v is None else f'{best_v:+.1%}'} | vault ok {r['passed_vault']} | "
                  f"ALL gates {r['passed_all']} | {r['seconds']}s", flush=True)
    path = mkt().data_dir / "search_width_test.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
