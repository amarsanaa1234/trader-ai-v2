"""Would stricter promotion checks have chosen better settings? A look-back test.

For several past cut-off dates: run the self-improver's search on the history
BEFORE the cut-off only, then let two rule sets pick (or refuse) a new setting:

  current  the market's present checks
  strict   + 40 trades per fold, 60 in the vault, shared-account vault profitable,
           beats 80% of random entries (FX's checks)

Each pick (or "keep the live setting" when nothing passes) is then traded on
the next `forward_days` after the cut-off, data no check ever saw.

    python tools/gate_experiment.py crypto [candidates] [forward_days] [months_back...]
"""
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

DAY = 86_400_000
STRICT = {"min_fold_trades": 40, "min_vault_trades": 60, "portfolio": True, "max_p_value": 0.20, "random_runs": 100}
_W = {}


def _init(market, cut):
    import markets
    from core import improver
    from core.context import activate
    from core.storage import Storage
    activate(markets.get(market))
    frames = improver.load_frames(Storage(), markets.get(market).symbols)
    _W["frames"] = {s: f[f.index < cut] for s, f in frames.items()}


def _wf(p):
    from core import validation
    return validation.walk_forward(_W["frames"], p)


def pick(frames, scored, cur_wf, cur_vault, rules):
    """The improver's promotion gates under a given rule set -> first setting that passes."""
    from core import improver, validation
    from core.context import mkt
    mkt().cfg["validation"] = dict(rules)
    try:
        for wf, cand in scored[:10]:
            if wf["score"] <= cur_wf["score"] or wf["positive_folds"] < improver.MIN_POSITIVE_FOLDS:
                continue
            v = validation.vault(frames, cand)
            if (v["score"] > cur_vault["score"] + improver.PROMOTE_MARGIN and v["trades"] >= improver.min_vault_trades()
                    and v["return"] > 0 and v["max_dd"] < improver.MAX_VAULT_DD):
                ok, extra = improver.extra_gates(frames, cand)
                if ok:
                    return cand, {"walk_forward": wf, "vault": v, **extra}
        return None, None
    finally:
        mkt().cfg["validation"] = BASE_RULES


def forward(full, p, cut, days):
    from core import portfolio
    r = portfolio.run(full, p, start_ts=cut, end_ts=cut + days * DAY, start_equity=10_000)
    return {k: r[k] for k in ("return", "max_dd", "trades", "avg_r")}


def main():
    global BASE_RULES
    import markets
    from core import improver, mistakes, validation
    from core.context import activate, mkt
    from core.storage import Storage
    from core.strategy import load_params

    market = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    months = [int(x) for x in sys.argv[4:]] or [6, 4, 2]
    activate(markets.get(market))
    BASE_RULES = dict(mkt().cfg.get("validation") or {})
    full = improver.load_frames(Storage(), mkt().symbols)
    end = min(int(f.index[-1]) for f in full.values())
    live = load_params()
    out = []
    for mb in months:
        cut = end - mb * 30 * DAY
        frames = {s: f[f.index < cut] for s, f in full.items()}
        rng = random.Random(1000 + mb)
        cands = [improver.mutate(live, 0.10, rng) for _ in range(int(n * 0.8))]
        cands += [improver.random_params(rng, live) for _ in range(n - len(cands))]
        with ProcessPoolExecutor(max_workers=16, initializer=_init, initargs=(market, cut)) as pool:
            results = list(pool.map(_wf, cands, chunksize=8))
        scored = sorted(zip(results, cands), key=lambda x: -x[0]["score"])
        for wf, c in list(scored[:3]):
            better, learned = mistakes.learn(frames, c, wf)
            if learned:
                scored.append((validation.walk_forward(frames, better), better))
        scored.sort(key=lambda x: -x[0]["score"])
        members = [{k: v for k, v in c.items() if k != "ensemble"} for _, c in scored[:5]]
        ens = {**members[0], "ensemble": members, "ensemble_votes": 3}
        scored.append((validation.walk_forward(frames, ens), ens))
        scored.sort(key=lambda x: -x[0]["score"])
        cur_wf, cur_vault = validation.walk_forward(frames, live), validation.vault(frames, live)

        row = {"cut_months_ago": mb, "forward_days": days, "live": forward(full, live, cut, days)}
        for name, rules in (("current", BASE_RULES), ("strict", {**BASE_RULES, **STRICT})):
            cand, why = pick(frames, scored, cur_wf, cur_vault, rules)
            row[name] = {"changed": cand is not None,
                         "forward": forward(full, cand, cut, days) if cand else row["live"],
                         "checks": why}
        out.append(row)
        f = lambda r: f"{r['forward']['return']:+.2%} (dd {r['forward']['max_dd']:.1%}, {r['forward']['trades']} tr)"
        print(f"{mb} сарын өмнө тасалсан: live {row['live']['return']:+.2%} | "
              f"одоогийн шалгалт {'ШИНЭ' if row['current']['changed'] else 'хэвээр'} {f(row['current'])} | "
              f"хатуу шалгалт {'ШИНЭ' if row['strict']['changed'] else 'хэвээр'} {f(row['strict'])}", flush=True)
    from core.settings import DATA_ROOT
    (DATA_ROOT / market / "gate_experiment.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
