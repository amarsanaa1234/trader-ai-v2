"""Classic chart / candlestick patterns, detected on closed candles only.

Each detector returns +1 (bullish), -1 (bearish) or 0. Whether a pattern is
actually worth anything is NOT assumed: measure_edges() checks how price moved
after each pattern on the training data, and only patterns with a real,
consistent edge get weight in the final score.
"""
import numpy as np
import pandas as pd

from .indicators import atr

PATTERNS = ["engulfing", "pin_bar", "double_top_bottom", "inside_bar_break", "structure", "three_soldiers_crows"]
EDGE_HORIZON = 12        # bars ahead used to judge a pattern
MIN_OCCURRENCES = 40


def _engulfing(df):
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)
    bull = (pc < po) & (c > o) & (c >= po) & (o <= pc)
    bear = (pc > po) & (c < o) & (c <= po) & (o >= pc)
    return bull.astype(int) - bear.astype(int)


def _pin_bar(df):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    hammer = (lower > 2 * body) & (lower / rng > 0.6)
    star = (upper > 2 * body) & (upper / rng > 0.6)
    return hammer.astype(int) - star.astype(int)


def _double_top_bottom(df, look=48, tol=0.004):
    """Price returns to a recent swing high (low) and closes back below (above)."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_high = h.shift(3).rolling(look).max()
    prev_low = l.shift(3).rolling(look).min()
    top = ((h - prev_high).abs() / prev_high < tol) & (c < h - (h - l) * 0.6)
    bottom = ((l - prev_low).abs() / prev_low < tol) & (c > l + (h - l) * 0.6)
    return bottom.astype(int) - top.astype(int)


def _inside_bar_break(df):
    h, l, c = df["high"], df["low"], df["close"]
    inside_prev = (h.shift(1) < h.shift(2)) & (l.shift(1) > l.shift(2))
    up = inside_prev & (c > h.shift(1))
    dn = inside_prev & (c < l.shift(1))
    return up.astype(int) - dn.astype(int)


def _structure(df, n=12):
    """Higher highs and higher lows vs. the previous window (and the reverse)."""
    h, l = df["high"], df["low"]
    hh = h.rolling(n).max() > h.shift(n).rolling(n).max()
    hl = l.rolling(n).min() > l.shift(n).rolling(n).min()
    lh = h.rolling(n).max() < h.shift(n).rolling(n).max()
    ll = l.rolling(n).min() < l.shift(n).rolling(n).min()
    return (hh & hl).astype(int) - (lh & ll).astype(int)


def _three_soldiers_crows(df):
    o, c = df["open"], df["close"]
    up = (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)) & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    dn = (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)) & (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    return up.astype(int) - dn.astype(int)


DETECTORS = {
    "engulfing": _engulfing, "pin_bar": _pin_bar, "double_top_bottom": _double_top_bottom,
    "inside_bar_break": _inside_bar_break, "structure": _structure, "three_soldiers_crows": _three_soldiers_crows,
}


def detect(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f"pat_{k}": fn(df).fillna(0).astype(float) for k, fn in DETECTORS.items()}, index=df.index)


def measure_edges(frames: dict, train_fraction: float) -> dict:
    """For each pattern: average forward move (in ATRs) in the signalled
    direction over the training data. Shrunk toward 0 when the sample is small
    or noisy, so random patterns end up near 0."""
    moves = {k: [] for k in PATTERNS}
    for df in frames.values():
        cut = int(len(df) * train_fraction)
        a = atr(df, 14)
        fwd = ((df["close"].shift(-EDGE_HORIZON) - df["close"]) / a).iloc[:cut]
        for k in PATTERNS:
            s = df[f"pat_{k}"].iloc[:cut]
            hit = s != 0
            moves[k] += list((fwd[hit] * s[hit]).dropna())
    edges = {}
    for k, m in moves.items():
        m = np.asarray(m)
        if len(m) < MIN_OCCURRENCES:
            edges[k] = 0.0
            continue
        t = m.mean() / (m.std(ddof=1) + 1e-9) * np.sqrt(len(m))
        # keep only patterns that are statistically distinguishable from noise
        edges[k] = round(float(np.clip(t / 4, -1, 1)), 3) if abs(t) > 2 else 0.0
    return edges


def pattern_score(df: pd.DataFrame, edges: dict) -> pd.Series:
    total = sum(df[f"pat_{k}"] * edges.get(k, 0.0) for k in PATTERNS)
    return np.tanh(total)
