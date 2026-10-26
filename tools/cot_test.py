"""Should the FX bot go AGAINST big speculators (COT) instead of with them? (owner, 2026-10-26)

tools/data_review.py found that over 10 years an extreme speculator position in a
currency was followed by the OPPOSITE move over the next 4 weeks (IC +0.11), while
the live FX settings give COT a positive weight (+0.63 = follow them). Three checks:

A. Is the raw signal stable?  The contrarian IC (next 5 days / 4 weeks) per year
   and per currency; data only from when each report was public (Friday 21:00 UTC).
B. Only the COT weight changed, everything else as live: +0.63 (now) ... -1.5.
C. Re-tuned: the improver's own search (mutations of the live settings), once with
   the COT weight free (as now) and once forced to be contrarian (<= 0); the best of
   each by walk-forward, then the vault, shared-account vault and random benchmark.

    python tools/cot_test.py [candidates]
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get("fx"))

from core import improver, portfolio, significance, validation  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params, load_params  # noqa: E402

PAIRS = {"EUR": ("EUR_USD", 1), "GBP": ("GBP_USD", 1), "AUD": ("AUD_USD", 1), "NZD": ("NZD_USD", 1),
         "JPY": ("USD_JPY", -1), "CAD": ("USD_CAD", -1), "CHF": ("USD_CHF", -1)}


def part_a(st):
    print("\n===== A. Raw COT signal: contrarian IC (speculators extreme -> the opposite move)")
    k = pd.read_sql_query("SELECT ccy, ts, net_pct FROM cot ORDER BY ts", st.db)
    rows = []
    for ccy, (sym, sign) in PAIRS.items():
        c = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='4h' ORDER BY ts", st.db,
                              params=(sym,)).set_index("ts")["close"]
        g = k[k.ccy == ccy].set_index("ts")["net_pct"]
        z = (g - g.rolling(156, min_periods=52).mean()) / g.rolling(156, min_periods=52).std()
        idx = np.searchsorted(c.index.to_numpy(), g.index.to_numpy())
        lc = np.log(c.to_numpy())
        for bars, lab in ((30, "5d"), (120, "4w")):
            ok = idx + bars < len(c)
            r = np.full(len(idx), np.nan)
            r[ok] = sign * (lc[idx[ok] + bars] - lc[idx[ok]])
            rows += [(ccy, pd.Timestamp(t, unit="ms").year, lab, -zz, rr) for t, zz, rr in zip(g.index, z, r)]
    d = pd.DataFrame(rows, columns=["ccy", "year", "h", "x", "y"]).dropna()

    def ic(g):
        return g["x"].corr(g["y"]) if len(g) > 10 else np.nan
    for lab in ("5d", "4w"):
        dd = d[d.h == lab]
        by_year = dd.groupby("year").apply(ic, include_groups=False)
        by_ccy = dd.groupby("ccy").apply(ic, include_groups=False)
        print(f"  next {lab}: all {ic(dd):+.3f} | by year " + " ".join(f"{y}:{v:+.2f}" for y, v in by_year.items()))
        print(f"  {'':9} by currency " + " ".join(f"{c}:{v:+.2f}" for c, v in by_ccy.items()) +
              f" | positive in {int((by_year > 0).sum())}/{len(by_year)} years, {int((by_ccy > 0).sum())}/{len(by_ccy)} currencies")


def judge(frames, p, n_random=100):
    wf, v = validation.walk_forward(frames, p), validation.vault(frames, p)
    pv = portfolio.vault(frames, p)
    sg = significance.random_benchmark(frames, p, n=n_random)
    return wf, v, pv, sg["p_value"]


def show(label, r):
    wf, v, pv, pval = r
    print(f"  {label:26} wf {wf['score']:+.2f} ({wf['positive_folds']}/4, {wf['trades']} tr, {wf['avg_r']:+.3f}R, "
          f"{' '.join(f'{x:+.1%}' for x in wf['fold_returns'])}) | vault {v['return']:+.2%} ({v['trades']} tr, "
          f"{v['avg_r']:+.3f}R) | one account {pv['return']:+.1%} dd {pv['max_dd']:.1%} | random p {pval:.2f}",
          flush=True)


def part_b(frames, live):
    print(f"\n===== B. Only the COT weight changed (live = {live['w_cot']:+.2f})")
    for w in (live["w_cot"], 0.3, 0.0, -0.3, -0.6, -0.9, -1.2, -1.5):
        show(f"w_cot {w:+.2f}" + ("  (live)" if w == live["w_cot"] else ""), judge(frames, clamp_params({**live, "w_cot": w})))


def part_c(frames, live, n):
    print(f"\n===== C. Re-tuned around the live settings ({n} candidates each, same random seed)")
    for label, force in (("COT free (as now)", False), ("COT contrarian (<= 0)", True)):
        rng = random.Random(42)
        cands = [improver.mutate(live, 0.15, rng) for _ in range(n)]
        if force:
            cands = [clamp_params({**c, "w_cot": -abs(c["w_cot"])}) for c in cands]
        for c in cands:
            c["pattern_edges"] = live.get("pattern_edges", {})
        res = improver.walk_forward_many(frames, cands)
        ranked = sorted(zip(res, cands), key=lambda x: -x[0]["score"])
        good = [(wf, c) for wf, c in ranked if wf["positive_folds"] >= improver.MIN_POSITIVE_FOLDS][:3]
        print(f"  {label}: best walk-forward {ranked[0][0]['score']:+.2f}; "
              f"{sum(r['positive_folds'] >= 3 for r in res)} of {n} have 3+ profitable windows")
        for i, (wf, c) in enumerate(good, 1):
            show(f"  #{i} w_cot {c['w_cot']:+.2f}", judge(frames, c))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    st = Storage()
    part_a(st)
    live = load_params()
    frames = improver.load_frames(st, mkt().symbols)
    part_b(frames, live)
    part_c(frames, live, n)


if __name__ == "__main__":
    main()
