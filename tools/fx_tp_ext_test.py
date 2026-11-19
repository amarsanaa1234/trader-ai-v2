"""FX 15m learner with the TP-extension rule (owner, 2026-11-19): core/orb15.TP_EXT variants
vs none, 5-year walk-forward from nothing, 0.5% risk, same as tools/orb_learn_test.py main().

    python tools/fx_tp_ext_test.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

VARIANTS = [None] + [{"tp_r": r, "reset_flat": f} for r in (1.5, 2.0, 3.0, 4.0) for f in (False, True)]


def _init(ext):
    from core import orb15
    orb15.TP_EXT = ext


def arms(ext):
    from core import orb_learn
    from orb_learn_test import arm_trades
    with ProcessPoolExecutor(10, initializer=_init, initargs=(ext,)) as ex:
        return [t for ts in ex.map(arm_trades, list(orb_learn.PAIRS)) for t in ts]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from core import orb15, orb_learn
    orb15.RISK, orb15.MAX_DD = 0.005, 0.99
    cut = int(pd.Timestamp("2024-10-28", tz="UTC").timestamp() * 1000)
    print(f"{'variant':24s} {'return':>8s} {'maxDD':>6s} {'risked':>6s} {'avg R':>7s} {'last 2y R':>9s} | all arms avg R")
    for ext in VARIANTS:
        cands = arms(ext)
        start = min(t["entry_ts"] for t in cands) + 30 * 86_400_000
        r = orb_learn.replay(cands, start)
        risked = [t for t in r["book"] if t["weight"] > 0 and "pnl" in t]
        rs = np.array([t["r"] for t in risked]) if risked else np.zeros(1)
        last2 = sum(t["r"] * t["weight"] for t in risked if t["entry_ts"] >= cut)
        name = "none (now)" if not ext else f"TP {ext['tp_r']}R{' +session reset' if ext['reset_flat'] else ''}"
        print(f"{name:24s} {r['return']:+8.1%} {r['max_dd']:6.1%} {len(risked):6d} {rs.mean():+7.3f} {last2:+9.1f} | "
              f"{np.mean([t['r'] for t in cands]):+.3f} ({len(cands)})", flush=True)


if __name__ == "__main__":
    main()
