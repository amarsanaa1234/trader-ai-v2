"""Candidate chart methods that are popular but unproven: Fibonacci retracements and
"Smart Money Concepts" (fair value gaps, liquidity sweeps, order blocks).

Each is written as a precise rule giving -1 / 0 / +1 per candle, using only
candles that had closed by then, so it can be tested like any other evidence
(tools/indicator_test.py). They are only used in trading if the test shows
they add something.
"""
import numpy as np
import pandas as pd

from .indicators import atr


def fibonacci(df: pd.DataFrame, look: int = 60, levels=(0.382, 0.5, 0.618), near_atr: float = 0.3) -> pd.Series:
    """Swing = highest high and lowest low of the last `look` candles. If the low came
    first (up-swing), a pullback that touches a 38.2/50/61.8% level and closes back
    above it = +1 (the swing continues). Mirror for down-swings = -1."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    a = atr(df, 14).to_numpy()
    out = np.zeros(len(df))
    for i in range(look, len(df)):
        win_h, win_l = h[i - look:i], l[i - look:i]
        hi_k, lo_k = int(np.argmax(win_h)), int(np.argmin(win_l))
        hi, lo = win_h[hi_k], win_l[lo_k]
        rng = hi - lo
        if rng <= 0 or not np.isfinite(a[i]):
            continue
        if lo_k < hi_k:   # up-swing: buy the pullback
            for r in levels:
                lvl = hi - r * rng
                if l[i] <= lvl + near_atr * a[i] and c[i] > lvl and l[i] >= lo:
                    out[i] = 1.0
                    break
        else:             # down-swing: sell the bounce
            for r in levels:
                lvl = lo + r * rng
                if h[i] >= lvl - near_atr * a[i] and c[i] < lvl and h[i] <= hi:
                    out[i] = -1.0
                    break
    return pd.Series(out, index=df.index)


def fair_value_gap(df: pd.DataFrame, max_age: int = 30) -> pd.Series:
    """Bullish gap: candle k's low is above candle k-2's high (price jumped, leaving a
    gap). When price later dips back into the still-open gap and closes inside or above
    it = +1. Mirror for bearish gaps = -1."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    out = np.zeros(len(df))
    bull, bear = [], []     # open gaps: (born, bottom, top)
    for i in range(2, len(df)):
        bull = [g for g in bull if i - g[0] <= max_age and l[i - 1] > g[1]]   # filled below bottom -> gone
        bear = [g for g in bear if i - g[0] <= max_age and h[i - 1] < g[2]]
        if any(l[i] <= top and c[i] >= bot for _, bot, top in bull):
            out[i] = 1.0
        elif any(h[i] >= bot and c[i] <= top for _, bot, top in bear):
            out[i] = -1.0
        if l[i] > h[i - 2]:
            bull.append((i, h[i - 2], l[i]))
        if h[i] < l[i - 2]:
            bear.append((i, h[i], l[i - 2]))
    return pd.Series(out, index=df.index)


def liquidity_sweep(df: pd.DataFrame, look: int = 20) -> pd.Series:
    """Price pokes above the last `look` candles' high (taking the stops resting there)
    but closes back below it = -1 (reversal down). Mirror below the low = +1."""
    prev_hi = df["high"].rolling(look).max().shift(1)
    prev_lo = df["low"].rolling(look).min().shift(1)
    up_sweep = (df["high"] > prev_hi) & (df["close"] < prev_hi)
    dn_sweep = (df["low"] < prev_lo) & (df["close"] > prev_lo)
    return pd.Series(np.where(dn_sweep, 1.0, np.where(up_sweep, -1.0, 0.0)), index=df.index)


def order_block(df: pd.DataFrame, impulse_atr: float = 2.0, bars: int = 3, max_age: int = 40) -> pd.Series:
    """Order block: the last opposite-colour candle before a strong move (> impulse_atr
    ATRs within `bars` candles). When price later comes back into that candle's range =
    a signal in the strong move's direction."""
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df, 14).to_numpy()
    out = np.zeros(len(df))
    blocks = []   # (born, side, low, high)
    for i in range(bars + 1, len(df)):
        blocks = [b for b in blocks if i - b[0] <= max_age]
        for b in blocks:
            if i > b[0] and l[i] <= b[3] and h[i] >= b[2]:
                out[i] = b[1]
                break
        if not np.isfinite(a[i]):
            continue
        move = c[i] - c[i - bars]
        if abs(move) > impulse_atr * a[i]:
            side = 1.0 if move > 0 else -1.0
            for k in range(i - bars, max(i - bars - 5, 0), -1):   # last opposite candle before the move
                if (c[k] < o[k]) if side > 0 else (c[k] > o[k]):
                    blocks.append((i, side, l[k], h[k]))
                    break
    return pd.Series(out, index=df.index)


def smc(df: pd.DataFrame) -> pd.Series:
    """The three Smart Money Concepts pieces together, in -1..1."""
    return (fair_value_gap(df) + liquidity_sweep(df) + order_block(df)).clip(-1, 1)


CANDIDATES = {"fibonacci": fibonacci, "fvg": fair_value_gap, "sweep": liquidity_sweep,
              "order_block": order_block, "smc": smc}
