"""Can tomorrow's FX direction be predicted better? (owner, 2026-11-11)

Target: the move over the next 24h (six 4h candles) in ATRs, for each of the 10 pairs.
Inputs (all known at the candle's close): every evidence source of the FX bot
(core/strategy.compute parts) plus the pair's own past moves (1, 5, 20 days in ATRs)
and volatility vs normal.

Walk-forward, never peeking: the model is refit at the start of every month on all
earlier candles (at least 2 years) and predicts that month only. Models:
  score     the live bot's own score (what it uses today)
  tsmom     sign of the pair's own 20-day move (the classic trend rule)
  ridge     linear model on all inputs (shrunk toward 0)
  gbm       small gradient-boosted trees on all inputs
Reported per year: rank IC (correlation of prediction with the 24h move) and hit rate
(direction right), on non-overlapping 24h steps so the counts are honest.

    python tools/dir24_test.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

AHEAD = 6
MIN_TRAIN_DAYS = 730


def dataset():
    import markets
    from core import library
    from core.context import activate
    from core.features import add_static_features
    from core.storage import Storage
    from core.strategy import compute, load_params, source_names
    m = markets.get("fx")
    activate(m)
    st, p = Storage(), load_params()
    parts = ["trend", "momentum", "breakout", "macd", "patterns", "htf", "mean_rev"] + \
        [s for s in source_names() if s not in ("trend", "momentum", "breakout", "macd", "patterns", "htf",
                                                 "mean_rev", "news", "library")]
    rows = []
    for sym in m.symbols:
        df = m.load_frame(st, sym)
        if len(df) < 3000:
            continue
        df = add_static_features(df, library.approved())
        out = compute(df, p)
        a = out["atr"]
        x = pd.DataFrame({k: np.asarray(out[k], dtype=float) for k in parts if k in out}, index=df.index)
        x["score"] = out["score"].to_numpy()
        for d in (1, 5, 20):
            x[f"ret{d}d"] = ((df["close"] - df["close"].shift(6 * d)) / a).clip(-10, 10)
        x["volr"] = a / a.rolling(6 * 60).mean()
        x["y"] = ((df["close"].shift(-AHEAD) - df["close"]) / a).clip(-8, 8)
        x["sym"] = sym
        rows.append(x)
    d = pd.concat(rows).replace([np.inf, -np.inf], np.nan)
    d["t"] = pd.to_datetime(d.index, unit="ms", utc=True)
    return d.dropna(subset=["y"]), [c for c in d.columns if c not in ("y", "sym", "t")]


def ic(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()) if ok.sum() > 30 else np.nan


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    d, feats = dataset()
    d = d.sort_values("t")
    months = pd.period_range(d.t.min() + pd.Timedelta(days=MIN_TRAIN_DAYS), d.t.max(), freq="M")
    preds = []
    for mo in months:
        start = mo.start_time.tz_localize("UTC")
        end = mo.end_time.tz_localize("UTC")
        train = d[d.t < start - pd.Timedelta(hours=4 * AHEAD)]          # its targets ended before the month
        test = d[(d.t >= start) & (d.t <= end)]
        if len(test) == 0:
            continue
        X, y = train[feats].fillna(0).to_numpy(), train["y"].to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-9
        r = Ridge(alpha=100.0).fit((X - mu) / sd, y)
        g = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=200,
                                          random_state=0).fit(X, y)
        Xt = test[feats].fillna(0).to_numpy()
        preds.append(test.assign(ridge=r.predict((Xt - mu) / sd), gbm=g.predict(Xt),
                                 tsmom=np.sign(test["ret20d"].to_numpy())))
    p = pd.concat(preds)
    p = p[(p.index // 3_600_000) % 24 == 0]          # one non-overlapping 24h step per pair per day
    print(f"{len(p)} out-of-sample pair-days, {p.t.min():%Y-%m} .. {p.t.max():%Y-%m}\n")
    print(f"{'year':6s} {'n':>5s} | " + " | ".join(f"{m:>14s}" for m in ("score", "tsmom", "ridge", "gbm")))
    for yr, g in list(p.groupby(p.t.dt.year)) + [("all", p)]:
        cells = []
        for m in ("score", "tsmom", "ridge", "gbm"):
            x, y = g[m].to_numpy(), g["y"].to_numpy()
            hit = np.mean(np.sign(x[x != 0]) == np.sign(y[x != 0]))
            cells.append(f"IC {ic(x, y):+.3f} {hit:4.0%}")
        print(f"{str(yr):6s} {len(g):5d} | " + " | ".join(f"{c:>14s}" for c in cells))
    # which inputs the linear model leans on (last fit)
    coef = pd.Series(r.coef_, index=feats).sort_values()
    print("\nridge weights (last refit, standardised):", {k: round(v, 3) for k, v in pd.concat([coef.head(4), coef.tail(4)]).items()})
    # strong signals only: top / bottom 20% of the ridge prediction
    q = p["ridge"].abs() >= p["ridge"].abs().quantile(0.8)
    print(f"ridge strongest 20%: hit {np.mean(np.sign(p.ridge[q]) == np.sign(p.y[q])):.1%} on {q.sum()} pair-days, "
          f"avg move with it {np.mean(np.sign(p.ridge[q]) * p.y[q]):+.3f} ATR")
    p[["sym", "t", "y", "score", "ridge", "gbm"]].to_parquet(Path(__file__).resolve().parent.parent / "data" / "fx" / "dir24_oos.parquet")


if __name__ == "__main__":
    main()
