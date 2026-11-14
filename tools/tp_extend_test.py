"""Owner's idea (2026-11-15): when a trade reaches its take-profit, ask the bot's own score
first. If the last closed candle still says "keep going" (the same signal that opened the
trade), do not take profit: lock the stop near the TP and push the TP further, again and
again while the score agrees. Otherwise take the TP as now.

Crypto, the live settings (data/crypto/strategy_params.json), one shared account
(core/portfolio.py), 2 years of 4h candles. Judged on the whole history, both halves
separately and the vault (newest 15%), next to "always extend" and "extend at random"
(same share of TPs) to see whether the score adds anything.

    python tools/tp_extend_test.py
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
    activate(markets.get("crypto"))
    st, strats = Storage(), library.approved()
    frames = {s: add_static_features(mkt().load_frame(st, s), strats) for s in mkt().symbols}
    base = strategy.load_params()
    feats = {s: strategy.compute(f, base) for s, f in frames.items()}
    t0 = max(int(f.index[300]) for f in frames.values())
    t1 = max(int(f.index[-1]) for f in frames.values())
    mid = (t0 + t1) // 2
    vault = max(int(df.index[research_end(len(df))]) for df in frames.values())
    print(f"{len(frames)} coins, live settings: TP {base['tp_r']:.2f}R, half at {base['partial_r']:.2f}R, "
          f"threshold {base['entry_threshold']}\n")

    variants = [("now (take the TP)", {})]
    for mode in ("strong", "half", "weak", "always"):
        for lock in (0.0, 0.5, 1.0):
            for ext in (1.0, 2.0):
                variants.append((f"{mode:6s} lock {lock:.1f}R ext +{ext:.0f}R",
                                 {"tp_extend": mode, "tp_ext_lock": lock, "tp_ext_r": ext}))
    for pct in (30, 60):
        variants.append((f"random {pct}%  lock 0.5R ext +1R",
                         {"tp_extend": "random", "tp_ext_pct": pct, "tp_ext_lock": 0.5, "tp_ext_r": 1.0}))

    print(f"{'variant':30s} {'2y return':>9s} {'DD':>6s} {'trades':>6s} {'avg R':>6s} | "
          f"{'1st half':>8s} {'2nd half':>8s} {'vault':>7s} | TPs extended")
    for name, extra in variants:
        p = {**base, **extra}
        a = portfolio.run(frames, p, start_ts=t0, feats=feats, with_curve=True)
        h1 = portfolio.run(frames, p, start_ts=t0, end_ts=mid, feats=feats)
        h2 = portfolio.run(frames, p, start_ts=mid, feats=feats)
        v = portfolio.run(frames, p, start_ts=vault, feats=feats)
        log = a["trade_log"]
        ex = [x["r"] for x in log if x["ext"]]
        tps = sum(1 for x in log if x["reason"] == "tp")
        exs = f"{len(ex)} trades avg {sum(ex) / len(ex):+.2f}R" if ex else "-"
        print(f"{name:30s} {a['return']:+9.1%} {a['max_dd']:6.1%} {a['trades']:6d} {a['avg_r']:+6.3f} | "
              f"{h1['return']:+8.1%} {h2['return']:+8.1%} {v['return']:+7.1%} | TP taken {tps}, extended {exs}",
              flush=True)


if __name__ == "__main__":
    main()
