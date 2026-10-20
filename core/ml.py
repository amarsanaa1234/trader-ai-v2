"""Machine-learning layer (research): LightGBM on triple-barrier labels.

Two ways to use a model, both judged with the SAME backtest engine, fees,
exits and risk guard as everything else:

  meta-labeling  the existing score decides long/short; the model only says
                 "take it" or "skip it" (probability the trade wins)
  direction      the model decides by itself: P(long wins) vs P(short wins)

No peeking: for every test window the model is trained only on candles that
ended at least `max_hold_bars` before the window starts (purged walk-forward),
and its probability threshold is picked on the last 20% of that training data.
"""
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import backtest, validation
from .indicators import adx, ema, rsi
from .reward import strategy_score
from .context import mkt
from .strategy import compute, source_names

log = logging.getLogger(__name__)
THRESHOLDS = np.arange(0.35, 0.725, 0.025)
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "feature_fraction": 0.7,
    "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 7,
    "deterministic": True, "num_threads": 8,
}
N_ROUNDS = 300


def features(df: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """Scale-free inputs, so one model can learn from all coins at once."""
    c = df["close"]
    a = f["atr"].replace(0, np.nan)
    parts = [k for k in source_names() if k not in ("news", "library")] + ["score", "regime"]
    X = pd.DataFrame({k: f[k] for k in parts}, index=df.index)
    X["volatile"] = pd.Series(f["volatile"], index=df.index).astype(float)
    X["atr_pct"] = a / c
    X["vol_rank"] = (a / c).rolling(720, min_periods=100).rank(pct=True)
    for k in (1, 4, 12, 24, 72):
        X[f"ret_{k}"] = (c - c.shift(k)) / a
    X["d_ema50"] = (c - ema(c, 50)) / a
    X["d_ema200"] = (c - ema(c, 200)) / a
    X["rsi14"] = rsi(c, 14) / 100
    X["d_hi24"] = (c - df["high"].rolling(24).max()) / a
    X["d_lo24"] = (c - df["low"].rolling(24).min()) / a
    X["vol_ratio"] = df["volume"] / df["volume"].rolling(24).mean()
    X["adx"] = adx(df, 14)
    for k in ("htf_1", "htf_2"):
        X[k] = df[k] if k in df else 0.0
    for col in mkt().rule_columns:  # raw market data, e.g. crypto funding rate
        X[f"{col}_raw"] = df[col]
    dt = pd.to_datetime(df.index, unit="ms", utc=True)
    X["hour"] = dt.hour
    X["dow"] = dt.dayofweek
    return X.astype(float)


def triple_barrier(df: pd.DataFrame, f: pd.DataFrame, p: dict, side: int) -> np.ndarray:
    """R result of entering at each candle's close with the bot's stop / take-profit /
    time limit: -1 if the stop is hit first, +tp_r if the target is, otherwise the
    move at the time limit. Fees included. NaN where the future isn't known yet."""
    c, h, l = (df[k].to_numpy() for k in ("close", "high", "low"))
    d = p["stop_atr"] * f["atr"].to_numpy()
    H, n = int(p["max_hold_bars"]), len(c)
    costs = mkt().costs
    fee_rt = costs["maker"] + costs["taker"]  # round trip: maker in, taker out (pessimistic)
    stop, tp = c - side * d, c + side * d * p["tp_r"]
    first_stop = np.full(n, H + 1)
    first_tp = np.full(n, H + 1)
    for j in range(1, H + 1):
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
        hh[: n - j], ll[: n - j] = h[j:], l[j:]
        hit_stop = (ll <= stop) if side == 1 else (hh >= stop)
        hit_tp = (hh >= tp) if side == 1 else (ll <= tp)
        first_stop = np.where((first_stop > H) & hit_stop, j, first_stop)
        first_tp = np.where((first_tp > H) & hit_tp, j, first_tp)
    r = np.where(first_stop <= first_tp, -1.0, p["tp_r"])  # stop wins ties (pessimistic)
    neither = (first_stop > H) & (first_tp > H)
    c_end = np.full(n, np.nan)
    c_end[: n - H] = c[H:]
    r = np.where(neither, side * (c_end - c) / d, r) - fee_rt * c / d
    r[n - H:] = np.nan
    r[~(d > 0)] = np.nan
    return r


def _fit(X: np.ndarray, y: np.ndarray):
    if len(y) < 150 or y.min() == y.max():
        return None
    params = dict(LGB_PARAMS, min_data_in_leaf=max(20, len(y) // 60))
    return lgb.train(params, lgb.Dataset(X, y), N_ROUNDS)


def _pick_threshold(prob: np.ndarray, r: np.ndarray, min_trades: int = 20) -> float:
    """Threshold that maximised total R on held-out training data; 1.0 (= never
    trade) if no threshold made money there."""
    best_t, best_sum = 1.0, 0.0
    for t in THRESHOLDS:
        take = prob >= t
        if take.sum() >= min_trades and r[take].sum() > best_sum:
            best_t, best_sum = float(t), float(r[take].sum())
    return best_t


class Dataset:
    """Features + labels for every coin of one timeframe."""

    def __init__(self, frames: dict, p: dict):
        self.p = p
        self.rows = {}
        idx = next(iter(frames.values())).index
        self.step = int(idx[1] - idx[0])
        for sym, df in frames.items():
            f = compute(df, p)
            self.rows[sym] = {
                "df": df, "atr": f["atr"], "primary": f["signal"].to_numpy(),
                "X": features(df, f).to_numpy(), "ts": df.index.to_numpy(),
                "r_long": triple_barrier(df, f, p, 1), "r_short": triple_barrier(df, f, p, -1),
            }

    def _collect(self, end_ts: int, mode: str, side: int = 0):
        """Training rows with candles before end_ts (labels fully known)."""
        Xs, rs, ts = [], [], []
        for d in self.rows.values():
            ok = d["ts"] < end_ts
            if mode == "meta":
                ok &= d["primary"] != 0
                r = np.where(d["primary"] == 1, d["r_long"], d["r_short"])
            else:
                r = d["r_long"] if side == 1 else d["r_short"]
            ok &= np.isfinite(r)
            Xs.append(d["X"][ok])
            rs.append(r[ok])
            ts.append(d["ts"][ok])
        return np.concatenate(Xs), np.concatenate(rs), np.concatenate(ts)

    def _train(self, end_ts: int, mode: str, side: int = 0):
        """Model + threshold using only data before end_ts."""
        X, r, ts = self._collect(end_ts, mode, side)
        if len(r) < 200:
            return None, 1.0
        cut = np.quantile(ts, 0.8)
        purge = int(self.p["max_hold_bars"]) * self.step
        inner = _fit(X[ts < cut - purge], (r[ts < cut - purge] > 0).astype(int))
        thr = _pick_threshold(inner.predict(X[ts >= cut]), r[ts >= cut]) if inner else 1.0
        return _fit(X, (r > 0).astype(int)), thr

    def signals(self, mode: str, start_ts: int, end_ts: int) -> dict:
        """Signal series per coin for candles in [start_ts, end_ts), models trained
        on data that ended `max_hold_bars` before start_ts."""
        purge = int(self.p["max_hold_bars"]) * self.step
        out, info = {}, {}
        if mode == "meta":
            model, thr = self._train(start_ts - purge, "meta")
            info = {"threshold": thr}
        elif mode == "direction":
            m_long, t_long = self._train(start_ts - purge, "direction", 1)
            m_short, t_short = self._train(start_ts - purge, "direction", -1)
            info = {"threshold_long": t_long, "threshold_short": t_short}
        for sym, d in self.rows.items():
            win = (d["ts"] >= start_ts) & (d["ts"] < end_ts)
            sig = np.zeros(len(d["ts"]), dtype=int)
            if mode == "base":
                sig[win] = d["primary"][win]
            elif mode == "meta":
                if model is not None and thr < 1.0:
                    prob = model.predict(d["X"][win])
                    sig[win] = np.where(prob >= thr, d["primary"][win], 0)
            else:
                pl = m_long.predict(d["X"][win]) if m_long is not None else np.zeros(win.sum())
                ps = m_short.predict(d["X"][win]) if m_short is not None else np.zeros(win.sum())
                edge_l, edge_s = pl - t_long, ps - t_short
                sig[win] = np.where((edge_l >= 0) & (edge_l >= edge_s), 1, np.where(edge_s >= 0, -1, 0))
            out[sym] = sig
        return out, info

    def evaluate(self, signals: dict, start_ts: int, end_ts: int) -> dict:
        rs, dds, rets, n = [], [], [], 0
        for sym, d in self.rows.items():
            idx = np.searchsorted(d["ts"], [start_ts, end_ts])
            feats = pd.DataFrame({"signal": signals[sym], "atr": d["atr"]}, index=d["df"].index)
            res = backtest.run(d["df"], self.p, start=int(idx[0]), end=int(idx[1]), feats=feats,
                               costs=mkt().costs_for(sym))
            rs += res.r
            dds.append(res.max_dd)
            rets.append(res.total_return)
            n += len(res.trades)
        dd = max(dds) if dds else 0.0
        return {"score": round(strategy_score(rs, dd, min_trades=validation.FOLD_MIN_TRADES), 3), "trades": n,
                "return": round(float(np.mean(rets)), 4), "max_dd": round(dd, 4),
                "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0}

    def windows(self):
        """Folds 2-4 (fold 1 has no earlier data to learn from) + the vault, as timestamps."""
        ts = next(iter(self.rows.values()))["ts"]
        folds, vault = validation.windows(len(ts))
        wins = [(f"fold{k + 1}", folds[k]) for k in range(1, len(folds))] + [("vault", vault)]
        return [(name, int(ts[a]), int(ts[b - 1]) + 1) for name, (a, b) in wins]

    def run(self, mode: str) -> dict:
        out = {}
        for name, a, b in self.windows():
            sig, info = self.signals(mode, a, b)
            out[name] = {**self.evaluate(sig, a, b), **info}
        folds = [v for k, v in out.items() if k != "vault"]
        out["summary"] = {"fold_score_mean": round(float(np.mean([f["score"] for f in folds])), 3),
                          "positive_folds": int(sum(f["return"] > 0 for f in folds)),
                          "vault_return": out["vault"]["return"], "vault_score": out["vault"]["score"],
                          "vault_avg_r": out["vault"]["avg_r"], "vault_trades": out["vault"]["trades"]}
        return out
