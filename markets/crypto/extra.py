"""Extra crypto evidence (added 2026-10-19), every value aligned to the moment a
candle closes so backtests never peek:

  premium   Coinbase (US buyers) vs Binance price gap, as a z-score of the last
            week. BTC/ETH use their own; other coins use BTC's (US demand for crypto).
  stocks    S&P 500 move over the last day (crypto trades like a risk asset)
  usd       US-dollar strength over the last day (a strong dollar weighs on crypto)

S&P 500 and the dollar come from the FX market's own database (OANDA), which
the FX bot keeps up to date; if it is missing these sources are simply 0.
"""
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from core.settings import DATA_ROOT
from core.storage import Storage

HOUR = 3_600_000
DAY = 24 * HOUR
FX_DB = DATA_ROOT / "fx" / "bot.db"
USD_PAIRS = {"EUR_USD": -1, "GBP_USD": -1, "AUD_USD": -1, "NZD_USD": -1, "USD_JPY": 1, "USD_CAD": 1, "USD_CHF": 1}


def _known_at(series: pd.Series, at: np.ndarray, max_age_ms: int | None = None) -> np.ndarray:
    """Latest value known at each time in `at` (series indexed by when each value became known)."""
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return np.full(len(at), np.nan)
    pos = np.searchsorted(series.index.to_numpy(), at, side="right") - 1
    vals = np.where(pos >= 0, series.to_numpy()[np.clip(pos, 0, None)], np.nan)
    if max_age_ms is not None:
        age = at - np.where(pos >= 0, series.index.to_numpy()[np.clip(pos, 0, None)], 0)
        vals = np.where(age <= max_age_ms, vals, 0.0)
    return vals


def premium_z(st, symbol: str, close: pd.Series, step_ms: int) -> np.ndarray:
    """z-score (last 7 days) of the Coinbase-vs-Binance premium, known at each candle's close."""
    coin = symbol.split("/")[0]
    product = f"{coin if coin in ('BTC', 'ETH') else 'BTC'}-USD"
    try:
        cb = pd.read_sql_query("SELECT ts, close FROM coinbase_candles WHERE product=? ORDER BY ts", st.db,
                               params=(product,)).set_index("ts")["close"]
    except Exception:
        return np.zeros(len(close))
    if cb.empty:
        return np.zeros(len(close))
    if product != f"{coin}-USD":   # alts: BTC's premium against Binance BTC
        ref = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol='BTC/USDT:USDT' AND tf='1h' ORDER BY ts",
                                st.db).set_index("ts")["close"]
    else:
        ref = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='1h' ORDER BY ts",
                                st.db, params=(symbol,)).set_index("ts")["close"]
    p = (cb.reindex(ref.index) / ref - 1).dropna()
    z = (p - p.rolling(168, min_periods=48).mean()) / p.rolling(168, min_periods=48).std()
    z.index = z.index + HOUR                       # an hourly candle is known when it closes
    return np.nan_to_num(_known_at(z.dropna(), close.index.to_numpy() + step_ms, max_age_ms=3 * HOUR))


def _fx_closes(symbol: str) -> pd.Series:
    if not FX_DB.exists():
        return pd.Series(dtype=float)
    db = sqlite3.connect(f"file:{FX_DB}?mode=ro", uri=True, timeout=30)
    try:
        return pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='4h' ORDER BY ts", db,
                                 params=(symbol,)).set_index("ts")["close"]
    except Exception:
        return pd.Series(dtype=float)
    finally:
        db.close()


def stocks_move(index: pd.Index, step_ms: int) -> np.ndarray:
    """S&P 500 log-move over the last 6 four-hour candles (~1 trading day); 0 when the
    stock market has been closed for a while (weekends)."""
    c = _fx_closes("SPX500_USD")
    if c.empty:
        return np.zeros(len(index))
    m = np.log(c).diff(6)
    m.index = m.index + 4 * HOUR
    return np.nan_to_num(_known_at(m.dropna(), index.to_numpy() + step_ms, max_age_ms=12 * HOUR))


def usd_move(index: pd.Index, step_ms: int) -> np.ndarray:
    """US-dollar basket log-move over the last day (+ = stronger dollar)."""
    parts = []
    for pair, side in USD_PAIRS.items():
        c = _fx_closes(pair)
        if not c.empty:
            parts.append(side * np.log(c).diff(6))
    if not parts:
        return np.zeros(len(index))
    m = pd.concat(parts, axis=1).mean(axis=1)
    m.index = m.index + 4 * HOUR
    return np.nan_to_num(_known_at(m.dropna(), index.to_numpy() + step_ms, max_age_ms=12 * HOUR))


# ---------------- Coinbase prices for the premium source ----------------
def _cb_table(st: Storage):
    st.db.execute("CREATE TABLE IF NOT EXISTS coinbase_candles "
                  "(product TEXT, ts INTEGER, close REAL, volume REAL, PRIMARY KEY(product, ts))")


def sync_coinbase(st: Storage, days_back: int = 730) -> int:
    """Hourly Coinbase BTC-USD / ETH-USD closes (public API, 300 candles per call)."""
    _cb_table(st)
    added = 0
    end = int(time.time()) // 3600 * 3600
    for coin in ("BTC", "ETH"):
        product = f"{coin}-USD"
        last = st.db.execute("SELECT MAX(ts) FROM coinbase_candles WHERE product=?", (product,)).fetchone()[0]
        t = (last // 1000 + 3600) if last else end - days_back * 86400
        while t < end:
            t2 = min(t + 300 * 3600, end)
            r = requests.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                             params={"granularity": 3600,
                                     "start": datetime.fromtimestamp(t, timezone.utc).isoformat(),
                                     "end": datetime.fromtimestamp(t2, timezone.utc).isoformat()},
                             headers={"User-Agent": "crypto_ai"}, timeout=20)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            rows = [(product, c[0] * 1000, c[4], c[5]) for c in r.json()]
            st.db.executemany("INSERT OR REPLACE INTO coinbase_candles VALUES (?, ?, ?, ?)", rows)
            st.db.commit()  # commit every chunk: never hold the shared DB lock while downloading
            added += len(rows)
            t = t2
            time.sleep(0.2)
    st.db.commit()
    return added
