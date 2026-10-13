"""Features that don't depend on tunable settings: higher-timeframe trend,
chart patterns and the strategy library's vote. Computed once per frame."""
import numpy as np
import pandas as pd

from . import library, patterns
from .indicators import atr, ema


def htf_trend(df: pd.DataFrame, rule: str, ema_n: int) -> np.ndarray:
    """Trend of a higher timeframe in [-1, 1], using only HTF candles that had
    fully closed when each base candle closed (no peeking)."""
    step = int(np.min(np.diff(df.index.to_numpy()[:200])))   # min: FX has weekend gaps
    dt = pd.to_datetime(df.index, unit="ms", utc=True)
    base = pd.DataFrame({"high": df["high"].to_numpy(), "low": df["low"].to_numpy(),
                         "close": df["close"].to_numpy()}, index=dt)
    agg = base.resample(rule, label="left", closed="left").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    val = np.tanh((agg["close"] - ema(agg["close"], ema_n)) / atr(agg, 14).replace(0, np.nan))
    val.index = val.index + pd.Timedelta(rule)            # when that HTF candle was complete
    close_times = dt + pd.Timedelta(milliseconds=step)    # when each base candle closes
    out = val.reindex(val.index.union(close_times)).ffill().reindex(close_times)
    return out.fillna(0).to_numpy()


def add_static_features(df: pd.DataFrame, strats: list[dict] | None = None) -> pd.DataFrame:
    if df.empty or len(df) < 2:
        return df
    df = pd.concat([df, patterns.detect(df)], axis=1)
    df["library"] = library.library_signal(df, strats)
    for k, (rule, n) in enumerate(htf_rules(), 1):
        df[f"htf_{k}"] = htf_trend(df, rule, int(n))
    return df


def htf_rules() -> list:
    """Two higher timeframes for the active market's base timeframe (config "htf"),
    default 4h + 1 day (for 1h candles)."""
    from .context import mkt
    m = mkt()
    rules = ((getattr(m, "cfg", None) or {}).get("htf") or {}).get(m.timeframe)
    return [tuple(r) for r in rules] if rules else [("4h", 50), ("1D", 20)]
