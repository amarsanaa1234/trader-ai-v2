"""Signal model: evidence sources scaled to [-1, 1], combined with learned
weights. Positive score = long, negative = short.

The core sources below work on any market (they only need candles). Each
market adds its own sources (crypto: funding, Fear & Greed, crowd
positioning; FX later: interest-rate carry, COT...) through
Market.sources(); their weights and tunable ranges come from the market too.

The self-improver tunes the numbers in param_space(). Risk limits are NOT here.
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .context import mkt
from .indicators import adx, atr, ema, rsi
from .patterns import pattern_score

CORE_SOURCES = ("trend", "momentum", "mean_rev", "breakout", "macd", "news", "patterns", "library", "htf")

CORE_PARAMS = {
    "ema_fast": 50, "ema_slow": 200, "rsi_len": 14, "atr_len": 14, "breakout_len": 24,
    "w_trend": 1.0,       # follow the EMA trend
    "w_momentum": 0.5,    # RSI momentum
    "w_breakout": 0.8,    # close beyond the recent high/low
    "w_macd": 0.4,        # MACD histogram
    "w_news": 0.5,        # AI news sentiment (live only; no history for backtests)
    "w_patterns": 0.5,    # chart patterns, each weighted by its measured edge
    "w_library": 0.6,     # vote of backtested strategies from the library
    "pattern_edges": {},  # set by the improver from historical data
    "entry_threshold": 1.3,
    "stop_atr": 2.0,      # stop distance in ATRs
    "tp_r": 2.0,          # take profit at this multiple of the stop distance
    "max_hold_bars": 48,
    # Entry: limit order this many ATRs better than the signal price (cheaper
    # maker fee + better price; if price never comes back, no trade).
    "entry_offset_atr": 0.1,
    # Market regime (ADX): trending vs ranging vs too volatile
    "adx_trend": 25, "adx_range": 18,
    "range_trend_scale": 0.3,  # in a range, trust trend/breakout signals this much
    "w_range_mr": 0.6,         # in a range, mean-reversion (fade RSI extremes)
    "vol_skip_pct": 0.97,      # skip entries when volatility is above this percentile (>=0.995 = off)
    # Higher timeframes (4h + 1d trend)
    "w_htf": 0.6,
    "htf_block": 0.6,          # never trade against a higher-timeframe trend stronger than this (>=0.99 = off)
    # Exits (see exits.py; below the "off" threshold = disabled)
    "be_at_r": 1.0,            # move stop to breakeven at +1R
    "trail_atr": 0.0,          # trailing stop distance in ATRs
    "partial_r": 1.0,          # take half the position off at +1R
    "allow_long": True,
    "allow_short": True,
}

# name: (low, high, is_int)
CORE_SPACE = {
    "ema_fast": (10, 100, True), "ema_slow": (100, 400, True), "rsi_len": (7, 28, True),
    "atr_len": (7, 28, True), "breakout_len": (6, 96, True),
    "w_trend": (0.0, 2.0, False), "w_momentum": (-1.0, 1.5, False), "w_breakout": (0.0, 2.0, False),
    "w_macd": (-1.0, 1.5, False), "w_patterns": (0.0, 2.0, False), "w_library": (0.0, 2.0, False),
    "entry_threshold": (0.6, 2.5, False),
    "stop_atr": (1.0, 4.0, False), "tp_r": (1.0, 4.0, False), "max_hold_bars": (6, 168, True),
    "entry_offset_atr": (0.0, 0.6, False),
    "adx_trend": (20, 40, True), "adx_range": (10, 22, True), "range_trend_scale": (0.0, 1.0, False),
    "w_range_mr": (0.0, 2.0, False), "vol_skip_pct": (0.8, 1.0, False),
    "w_htf": (0.0, 2.0, False), "htf_block": (0.2, 1.0, False),
    "be_at_r": (0.0, 2.5, False), "trail_atr": (0.0, 4.0, False), "partial_r": (0.0, 3.0, False),
}


def default_params() -> dict:
    return {**CORE_PARAMS, **mkt().default_params}


def param_space() -> dict:
    return {**CORE_SPACE, **mkt().param_space}


def source_names() -> tuple:
    """Every evidence source in the score, core first, then the market's own."""
    return CORE_SOURCES + tuple(mkt().source_names)


def weight_key(source: str) -> str:
    return "w_range_mr" if source == "mean_rev" else f"w_{source}"


def clamp_params(p: dict) -> dict:
    space = param_space()
    out = {**default_params(), **p}
    for k, (lo, hi, is_int) in space.items():
        v = min(max(float(out[k]), lo), hi)
        out[k] = int(round(v)) if is_int else round(v, 6)
    if out["ema_fast"] >= out["ema_slow"]:
        out["ema_fast"] = max(space["ema_fast"][0], out["ema_slow"] // 3)
    out["adx_range"] = min(out["adx_range"], out["adx_trend"] - 2)
    return out


def load_params() -> dict:
    path = mkt().params_path
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return clamp_params(data.get("params", {}))
    return default_params()


def vault_score() -> float:
    """Score of the live settings on the never-ranked newest history (for the live-mode gate)."""
    path = mkt().params_path
    if not path.exists():
        return -99.0
    m = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
    v = m.get("vault") or m.get("backtest", {}).get("vault") or {}
    return float(v.get("score", -99.0))


def save_params(params: dict, metrics: dict, note: str) -> None:
    path, hist = mkt().params_path, mkt().params_history_dir
    hist.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if path.exists():
        (hist / f"params_{stamp}.json").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps({
        "updated_utc": stamp, "note": note, "params": clamp_params(params), "metrics": metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def compute(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Adds component columns, score, atr and signal (+1 long / -1 short / 0 flat)."""
    c = df["close"]
    a = atr(df, int(p["atr_len"]))
    ef, es = ema(c, int(p["ema_fast"])), ema(c, int(p["ema_slow"]))
    n = int(p["breakout_len"])
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)

    out = pd.DataFrame(index=df.index)
    out["trend"] = np.tanh((ef - es) / a.replace(0, np.nan)).fillna(0)
    out["momentum"] = ((rsi(c, int(p["rsi_len"])) - 50) / 25).clip(-1, 1)
    out["breakout"] = np.where(c > hi, 1.0, np.where(c < lo, -1.0, 0.0))
    out["news"] = df["news"].clip(-1, 1) if "news" in df else 0.0
    macd = ema(c, 12) - ema(c, 26)
    out["macd"] = np.tanh((macd - ema(macd, 9)) / a.replace(0, np.nan)).fillna(0)
    out["patterns"] = pattern_score(df, p.get("pattern_edges") or {}) if "pat_engulfing" in df else 0.0
    out["library"] = df["library"] if "library" in df else 0.0
    out["htf"] = ((df["htf_4h"] + df["htf_1d"]) / 2).fillna(0) if "htf_4h" in df else 0.0
    market_parts = mkt().sources(df, p)
    for name in mkt().source_names:
        out[name] = market_parts[name]

    # Regime: +1 trending, -1 ranging, 0 in between. In a range, trend-following
    # evidence is discounted and fading RSI extremes (mean reversion) is added.
    strength = adx(df, 14)
    out["regime"] = np.where(strength >= p["adx_trend"], 1, np.where(strength <= p["adx_range"], -1, 0))
    ranging = out["regime"] == -1
    trend_scale = np.where(ranging, p["range_trend_scale"], 1.0)
    out["mean_rev"] = np.where(ranging, -out["momentum"], 0.0)
    vol_rank = (a / c).rolling(720, min_periods=100).rank(pct=True).fillna(0.5)
    out["volatile"] = (vol_rank > p["vol_skip_pct"]) if p["vol_skip_pct"] < 0.995 else False

    score = trend_scale * (p["w_trend"] * out["trend"] + p["w_breakout"] * out["breakout"])
    for name in source_names():
        if name not in ("trend", "breakout"):
            score = score + p[weight_key(name)] * out[name]
    out["score"] = score
    out["atr"] = a
    thr = p["entry_threshold"]
    long_ok = (out["score"] >= thr) & p["allow_long"]
    short_ok = (out["score"] <= -thr) & p["allow_short"]
    if p["htf_block"] < 0.99:  # don't fight a strong higher-timeframe trend
        long_ok &= out["htf"] > -p["htf_block"]
        short_ok &= out["htf"] < p["htf_block"]
    sig = np.where(long_ok, 1, np.where(short_ok, -1, 0))
    sig[np.asarray(out["volatile"], dtype=bool)] = 0
    sig[: int(p["ema_slow"])] = 0  # indicator warm-up
    out["signal"] = sig
    return out
