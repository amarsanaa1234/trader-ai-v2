"""Walk-forward validation.

History is split into 4 consecutive "fold" windows plus a final vault:

  |  fold 1  |  fold 2  |  fold 3  |  fold 4  | vault |
  0%                                        85%    100%

Settings are ranked by how well AND how consistently they do across the 4
folds (mean minus spread), so something that only worked in one period loses.
The vault is only used as a final pass/fail gate, never for ranking.
"""
import numpy as np

from . import backtest
from .reward import strategy_score
from .strategy import compute

VAULT_FRACTION = 0.15
N_FOLDS = 4
FOLD_MIN_TRADES = 8   # default; a market may demand more (config validation.min_fold_trades)


def min_trades() -> int:
    from .context import mkt
    return int(((getattr(mkt(), "cfg", None) or {}).get("validation") or {}).get("min_fold_trades", FOLD_MIN_TRADES))


def windows(n_rows: int):
    cut = int(n_rows * (1 - VAULT_FRACTION))
    edges = np.linspace(0, cut, N_FOLDS + 1).astype(int)
    folds = [(int(edges[k]), int(edges[k + 1])) for k in range(N_FOLDS)]
    return folds, (cut, n_rows)


def research_end(n_rows: int) -> int:
    """Rows before this index may be used for fitting (pattern edges etc.)."""
    return int(n_rows * (1 - VAULT_FRACTION))


def _window_stats(frames, p, pick, feats_cache=None, signal_fn=None):
    rs, dds, rets, n = [], [], [], 0
    for sym, df in frames.items():
        feats = signal_fn(df) if signal_fn else (feats_cache or {}).get(sym)
        if feats is None:
            feats = compute(df, p)
        start, end = pick(len(df))
        res = backtest.run(df, p, start=start, end=end, feats=feats)
        rs += res.r
        dds.append(res.max_dd)
        rets.append(res.total_return)
        n += len(res.trades)
    return rs, (max(dds) if dds else 0.0), (float(np.mean(rets)) if rets else 0.0), n


def walk_forward(frames: dict, p: dict, signal_fn=None) -> dict:
    feats = None if signal_fn else {s: compute(df, p) for s, df in frames.items()}
    fold_scores, fold_returns, total, all_r = [], [], 0, []
    for k in range(N_FOLDS):
        rs, dd, ret, n = _window_stats(frames, p, lambda m, k=k: windows(m)[0][k], feats, signal_fn)
        fold_scores.append(round(strategy_score(rs, dd, min_trades=min_trades()), 3))
        fold_returns.append(round(ret, 4))
        total += n
        all_r += rs
    fs = np.asarray(fold_scores)
    return {"score": round(float(fs.mean() - 0.5 * fs.std()), 3), "folds": fold_scores, "fold_returns": fold_returns,
            "positive_folds": int(sum(r > 0 for r in fold_returns)), "trades": total,
            "avg_r": round(float(np.mean(all_r)), 3) if all_r else 0.0}


def vault(frames: dict, p: dict, signal_fn=None) -> dict:
    feats = None if signal_fn else {s: compute(df, p) for s, df in frames.items()}
    rs, dd, ret, n = _window_stats(frames, p, lambda m: windows(m)[1], feats, signal_fn)
    return {"score": round(strategy_score(rs, dd, min_trades=min_trades()), 3), "trades": n,
            "return": round(ret, 4), "max_dd": round(dd, 4),
            "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0}


def forward(frames: dict, p: dict, since_ms: int) -> dict:
    """Performance on candles after `since_ms` only (used by the shadow test)."""
    rs, dds, rets, n = [], [], [], 0
    for df in frames.values():
        start = int(np.searchsorted(df.index.to_numpy(), since_ms))
        if start >= len(df) - 1:
            continue
        res = backtest.run(df, p, start=start, feats=compute(df, p))
        rs += res.r
        dds.append(res.max_dd)
        rets.append(res.total_return)
        n += len(res.trades)
    dd = max(dds) if dds else 0.0
    # a shadow only sees a few weeks of new candles, so the market's high minimum doesn't apply
    return {"score": round(strategy_score(rs, dd, min_trades=FOLD_MIN_TRADES), 3), "trades": n,
            "return": round(float(np.mean(rets)), 4) if rets else 0.0, "max_dd": round(dd, 4),
            "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0}
