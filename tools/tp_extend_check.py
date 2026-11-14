"""Before/after the fixed TP-extension rule (owner, 2026-11-15) on crypto and commodities:
live settings, one shared account, whole history / halves / vault. Flips the market's
tp_extend flag in memory only.

    python tools/tp_extend_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")


def main():
    import markets
    from core import library, portfolio, strategy
    from core.context import activate, mkt
    from core.features import add_static_features
    from core.storage import Storage
    from core.validation import research_end
    for name in sys.argv[1:] or ["crypto", "commodities"]:
        activate(markets.get(name))
        st, strats = Storage(), library.approved()
        frames = {s: add_static_features(mkt().load_frame(st, s), strats) for s in mkt().symbols}
        frames = {s: f for s, f in frames.items() if len(f) > 1000}
        p = strategy.load_params()
        feats = {s: strategy.compute(f, p) for s, f in frames.items()}
        t0 = max(int(f.index[300]) for f in frames.values())
        t1 = max(int(f.index[-1]) for f in frames.values())
        mid, vault = (t0 + t1) // 2, max(int(df.index[research_end(len(df))]) for df in frames.values())
        print(f"\n== {name}: {len(frames)} symbols")
        for on in (False, True):
            mkt().cfg["tp_extend"] = on
            a = portfolio.run(frames, p, start_ts=t0, feats=feats, with_curve=True)
            h1 = portfolio.run(frames, p, start_ts=t0, end_ts=mid, feats=feats)
            h2 = portfolio.run(frames, p, start_ts=mid, feats=feats)
            v = portfolio.run(frames, p, start_ts=vault, feats=feats)
            print(f"{'with rule' if on else 'before':10s} all {a['return']:+8.1%} DD {a['max_dd']:5.1%} "
                  f"{a['trades']:5d} tr {a['avg_r']:+.3f}R | halves {h1['return']:+7.1%} {h2['return']:+7.1%} "
                  f"| vault {v['return']:+6.1%} DD {v['max_dd']:5.1%}", flush=True)


if __name__ == "__main__":
    main()
