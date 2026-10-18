"""Is the edge real, or luck from testing a thousand settings?

The settings' trades on the vault (never used for ranking) are compared with
`n` runs of RANDOM entries: the same number of signals per symbol, the same
long/short mix, the same stops, targets, costs and blocked hours - only the
timing is random. p-value = share of random runs that did at least as well
(average R). A real edge should beat ~90%+ of them (p <= 0.10).
"""
import numpy as np

from . import backtest, validation
from .context import mkt
from .strategy import compute


def random_benchmark(frames: dict, p: dict, n: int = 200, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    base = {}
    actual = []
    for sym, df in frames.items():
        feats = compute(df, p)
        start = validation.research_end(len(df))
        res = backtest.run(df, p, start=start, feats=feats, costs=mkt().costs_for(sym))
        actual += res.r
        window = np.arange(max(start - 1, int(p["ema_slow"])), len(df) - 1)
        allowed = window[~np.asarray(mkt().no_entry(df.index), dtype=bool)[window]]
        k = len(res.trades) + res.missed              # entry attempts, not signal candles
        longs = float(np.mean([t["side"] == 1 for t in res.trades])) if res.trades else 0.5
        base[sym] = (df, feats, start, allowed, k, longs)
    if not actual:
        return {"trades": 0, "avg_r": 0.0, "p_value": 1.0, "random_avg_r": 0.0, "runs": 0}
    real = float(np.mean(actual))
    rand = []
    for _ in range(n):
        rs = []
        for sym, (df, feats, start, allowed, k, longs) in base.items():
            if k == 0 or len(allowed) == 0:
                continue
            f = feats.copy()
            s = np.zeros(len(df), dtype=int)
            pick = rng.choice(allowed, size=min(k, len(allowed)), replace=False)
            s[pick] = np.where(rng.random(len(pick)) < longs, 1, -1)
            f["signal"] = s
            rs += backtest.run(df, p, start=start, feats=f, costs=mkt().costs_for(sym)).r
        rand.append(float(np.mean(rs)) if rs else 0.0)
    rand = np.asarray(rand)
    return {"trades": len(actual), "avg_r": round(real, 3), "runs": n,
            "random_avg_r": round(float(rand.mean()), 3), "random_p90": round(float(np.quantile(rand, 0.9)), 3),
            "p_value": round(float((1 + (rand >= real).sum()) / (n + 1)), 3)}
