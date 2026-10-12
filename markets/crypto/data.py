"""Crypto market data from public endpoints (no API key): candles, funding rates,
crowd positioning, Fear & Greed.

Signals are always computed from real Binance (mainnet) data, even in demo
mode, so what the bot learns matches the real market.
"""
import logging

import ccxt
import numpy as np
import pandas as pd
import requests

from core.storage import Storage, now_ms

log = logging.getLogger(__name__)
HOUR_MS = 3_600_000
DAY_MS = 86_400_000

_public = None


def public_exchange():
    global _public
    if _public is None:
        _public = ccxt.binanceusdm({"enableRateLimit": True})
        _public.load_markets()
    return _public


SCHEMA = """
CREATE TABLE IF NOT EXISTS funding (symbol TEXT, ts INTEGER, rate REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS fng (ts INTEGER PRIMARY KEY, value REAL);
CREATE TABLE IF NOT EXISTS positioning (
    symbol TEXT, ts INTEGER, global_ls REAL, top_ls REAL, oi_value REAL, PRIMARY KEY (symbol, ts));
"""


def tf_ms(tf: str) -> int:
    return public_exchange().parse_timeframe(tf) * 1000


def sync_candles(st: Storage, symbol: str, tf: str, days: int) -> int:
    ex = public_exchange()
    step = tf_ms(tf)
    last = st.last_candle_ts(symbol, tf)
    since = (last + step) if last else now_ms() - days * DAY_MS
    added = 0
    while True:
        rows = ex.fetch_ohlcv(symbol, tf, since=since, limit=1500)
        # keep only fully closed candles
        cutoff = now_ms() - step
        rows = [r for r in rows if r[0] <= cutoff]
        if not rows:
            break
        st.insert_candles(symbol, tf, rows)
        added += len(rows)
        since = rows[-1][0] + step
        if since > cutoff:
            break
    return added


def sync_funding(st: Storage, symbol: str, days: int) -> int:
    ex = public_exchange()
    last = st.db.execute("SELECT MAX(ts) m FROM funding WHERE symbol=?", (symbol,)).fetchone()["m"]
    since = (last + 1) if last else now_ms() - days * DAY_MS
    added = 0
    while True:
        rows = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        rows = [r for r in rows if r["timestamp"] >= since]
        if not rows:
            break
        # Binance stamps funding a few ms after the hour; round to the hour.
        st.db.executemany("INSERT OR REPLACE INTO funding VALUES (?, ?, ?)",
                          [(symbol, (r["timestamp"] // HOUR_MS) * HOUR_MS, r["fundingRate"]) for r in rows])
        st.db.commit()
        added += len(rows)
        since = rows[-1]["timestamp"] + 1
        if len(rows) < 1000:
            break
    return added


def sync_fng(st: Storage) -> int:
    """Crypto Fear & Greed index (daily, 0 = extreme fear, 100 = extreme greed)."""
    resp = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=20)
    resp.raise_for_status()
    rows = [(int(d["timestamp"]) * 1000, float(d["value"])) for d in resp.json()["data"]]
    st.db.executemany("INSERT OR REPLACE INTO fng VALUES (?, ?)", rows)
    st.db.commit()
    return len(rows)


def sync_positioning(st: Storage, symbol: str) -> int:
    """Crowd positioning from Binance: long/short account ratio, top-trader
    position ratio, open interest. Binance only serves the last 30 days, so the
    bot keeps collecting it every hour and builds its own history over time."""
    ex = public_exchange()
    mid = ex.market(symbol)["id"]
    start = now_ms() - 29 * DAY_MS
    sources = [
        ("global_ls", ex.fapidata_get_globallongshortaccountratio, "longShortRatio"),
        ("top_ls", ex.fapidata_get_toplongshortpositionratio, "longShortRatio"),
        ("oi_value", ex.fapidata_get_openinteresthist, "sumOpenInterestValue"),
    ]
    total = 0
    for column, fn, field in sources:
        since = start
        while since < now_ms() - HOUR_MS:
            rows = fn({"symbol": mid, "period": "1h", "limit": 500, "startTime": since})
            if not rows:
                break
            st.db.executemany(
                f"INSERT INTO positioning(symbol, ts, {column}) VALUES (?, ?, ?) "
                f"ON CONFLICT(symbol, ts) DO UPDATE SET {column}=excluded.{column}",
                [(symbol, (int(r["timestamp"]) // HOUR_MS) * HOUR_MS, float(r[field])) for r in rows])
            st.db.commit()
            total += len(rows)
            since = int(rows[-1]["timestamp"]) + HOUR_MS
            if len(rows) < 500:
                break
    return total


def sync_all(st: Storage, symbols, tf: str, days: int) -> None:
    for s in symbols:
        n = sync_candles(st, s, tf, days)
        f = sync_funding(st, s, days)
        try:
            pos = sync_positioning(st, s)
        except Exception as e:
            pos = 0
            log.warning("%s: long/short өгөгдөл татаж чадсангүй: %s", s, e)
        log.info("%s: +%d лаа, +%d funding, +%d positioning", s, n, f, pos)
    try:
        sync_fng(st)
    except Exception as e:  # optional data source
        log.warning("Fear&Greed татаж чадсангүй: %s", e)


def load_frame(st: Storage, symbol: str, tf: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND tf=? ORDER BY ts",
        st.db, params=(symbol, tf)).set_index("ts")
    if df.empty:
        return df

    fund = pd.read_sql_query("SELECT ts, rate FROM funding WHERE symbol=? ORDER BY ts", st.db,
                             params=(symbol,)).drop_duplicates("ts").set_index("ts")["rate"]
    # hold_cost: the funding rate actually charged on this bar (0 if no settlement)
    df["hold_cost"] = fund.reindex(df.index).fillna(0.0)
    # funding: latest known rate, used as a crowding feature
    df["funding"] = fund.reindex(df.index.union(fund.index)).ffill().reindex(df.index).fillna(0.0)

    fng = pd.read_sql_query("SELECT ts, value FROM fng ORDER BY ts", st.db).set_index("ts")["value"]
    if fng.empty:
        df["fng"] = 50.0
    else:
        day = (df.index // DAY_MS) * DAY_MS
        # yesterday's value, so a day's index is never used before it is published
        fng_shift = fng.copy()
        fng_shift.index = fng_shift.index + DAY_MS
        vals = fng_shift.reindex(fng_shift.index.union(pd.Index(day.unique()))).ffill()
        df["fng"] = np.asarray(vals.reindex(day).fillna(50.0))
    pos = pd.read_sql_query("SELECT ts, global_ls, top_ls, oi_value FROM positioning WHERE symbol=? ORDER BY ts",
                            st.db, params=(symbol,)).set_index("ts")
    for col in ("global_ls", "top_ls", "oi_value"):
        df[col] = pos[col].reindex(df.index) if not pos.empty else np.nan
    df["news"] = 0.0
    return df
