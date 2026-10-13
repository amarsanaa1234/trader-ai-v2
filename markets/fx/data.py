"""FX market data from OANDA: mid candles, OANDA clients' positioning (position
book), financing (swap) rates, and a broad USD-strength series built from the
traded pairs."""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from core.storage import Storage, now_ms

from .oanda import OandaError, data_client

log = logging.getLogger(__name__)
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
GRAN = {"5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D"}
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": DAY_MS}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SCHEMA = """
CREATE TABLE IF NOT EXISTS book (symbol TEXT, ts INTEGER, long_pct REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS financing (
    symbol TEXT PRIMARY KEY, long_rate REAL, short_rate REAL, days TEXT, updated INTEGER);
"""


def usd_side(symbol: str) -> int:
    """+1 if USD is the base (USD_JPY rises when USD strengthens), -1 if it is the quote, 0 if absent."""
    base, quote = symbol.split("_")
    return 1 if base == "USD" else -1 if quote == "USD" else 0


# ---------- sync ----------
def sync_candles(st: Storage, symbol: str, tf: str, days: int) -> int:
    c = data_client()
    step = TF_MS[tf]
    last = st.last_candle_ts(symbol, tf)
    since = (last + step) if last else now_ms() - days * DAY_MS
    added = 0
    while since < now_ms() - step:
        js = c.get(f"/instruments/{symbol}/candles", granularity=GRAN[tf], price="M",
                   **{"from": f"{since / 1000:.0f}", "count": 5000})
        rows = [(int(float(k["time"]) * 1000), float(k["mid"]["o"]), float(k["mid"]["h"]), float(k["mid"]["l"]),
                 float(k["mid"]["c"]), float(k["volume"])) for k in js["candles"] if k["complete"]]
        rows = [r for r in rows if r[0] >= since]
        if not rows:
            break
        st.insert_candles(symbol, tf, rows)
        added += len(rows)
        since = rows[-1][0] + step
        if len(js["candles"]) < 5000:
            break
    return added


def _book_at(symbol: str, ts_s: int | None):
    params = {"time": str(ts_s)} if ts_s else {}
    try:
        b = data_client().get(f"/instruments/{symbol}/positionBook", **params)["positionBook"]
    except OandaError as e:
        log.debug("%s position book %s: %s", symbol, ts_s, e)
        return None
    long_pct = sum(float(x["longCountPercent"]) for x in b["buckets"])
    short_pct = sum(float(x["shortCountPercent"]) for x in b["buckets"])
    if long_pct + short_pct <= 0:
        return None
    ts = b.get("unixTime") or pd.Timestamp(b["time"]).timestamp()   # the book ignores the UNIX header
    return int(float(ts) * 1000), 100 * long_pct / (long_pct + short_pct)


def sync_book(st: Storage, symbol: str, days: int, every_hours: int = 6) -> int:
    """OANDA keeps position-book snapshots every 20 minutes back for years; history is
    filled at `every_hours` spacing, then the newest snapshot is added every hour."""
    step = every_hours * HOUR_MS
    last = st.db.execute("SELECT MAX(ts) m FROM book WHERE symbol=?", (symbol,)).fetchone()["m"]
    start = max(last + step if last else 0, now_ms() - days * DAY_MS)
    start = (start // step + 1) * step
    times = list(range(start, now_ms() - step, step))
    rows = []
    if times:
        with ThreadPoolExecutor(8) as pool:
            rows = [r for r in pool.map(lambda t: _book_at(symbol, t // 1000), times) if r]
    now = _book_at(symbol, None)
    if now:
        rows.append(now)
    st.db.executemany("INSERT OR REPLACE INTO book VALUES (?, ?, ?)", [(symbol, ts, p) for ts, p in rows])
    st.db.commit()
    return len(rows)


def sync_financing(st: Storage, symbols) -> int:
    c = data_client()
    if not c.account:
        return 0
    js = c.get(c.acct("/instruments"), instruments=",".join(symbols))
    rows = []
    for i in js["instruments"]:
        f = i.get("financing") or {}
        days = {d["dayOfWeek"]: int(d["daysCharged"]) for d in f.get("financingDaysOfWeek", [])}
        rows.append((i["name"], float(f.get("longRate", 0)), float(f.get("shortRate", 0)),
                     ",".join(str(days.get(w, 1 if w not in ("Saturday", "Sunday") else 0)) for w in WEEKDAYS),
                     now_ms()))
    st.db.executemany("INSERT OR REPLACE INTO financing VALUES (?, ?, ?, ?, ?)", rows)
    st.db.commit()
    return len(rows)


# ---------- frame ----------
def _hold_cost(st: Storage, symbol: str, index: pd.Index, tf: str) -> np.ndarray:
    """Swap charged at the 17:00 New York rollover, as a symmetric long cost:
    long pays h, short pays -h, h = (shortRate - longRate) / 2 per year. (Both sides
    also pay OANDA's markup, about 1%/year on majors; small next to the spread.)"""
    row = st.db.execute("SELECT long_rate, short_rate, days FROM financing WHERE symbol=?", (symbol,)).fetchone()
    if not row:
        return np.zeros(len(index))
    h_day = (row["short_rate"] - row["long_rate"]) / 2 / 365
    days = [int(x) for x in row["days"].split(",")]
    opened = pd.to_datetime(index, unit="ms", utc=True).tz_convert("America/New_York")
    closed = opened + pd.Timedelta(milliseconds=TF_MS[tf])
    roll = closed.normalize() + pd.Timedelta(hours=17)
    charged = (opened < roll) & (roll <= closed)
    mult = np.array([days[d] for d in roll.weekday])
    return np.where(charged, h_day * mult, 0.0)


def _usd_momentum(st: Storage, tf: str, symbols, index: pd.Index, bars: int = 24) -> np.ndarray:
    """Broad US-dollar strength: average of each pair's last-24-bar move, turned so +
    means USD got stronger."""
    parts = []
    for s in symbols:
        side = usd_side(s)
        if not side:
            continue
        c = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf=? ORDER BY ts", st.db,
                              params=(s, tf)).set_index("ts")["close"]
        parts.append(side * np.log(c).diff(bars))
    if not parts:
        return np.zeros(len(index))
    usd = pd.concat(parts, axis=1).mean(axis=1)
    return usd.reindex(index).ffill().fillna(0.0).to_numpy()


def load_frame(st: Storage, symbol: str, tf: str, symbols) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND tf=? ORDER BY ts",
        st.db, params=(symbol, tf)).set_index("ts")
    if df.empty:
        return df
    df["hold_cost"] = _hold_cost(st, symbol, df.index, tf)
    book = pd.read_sql_query("SELECT ts, long_pct FROM book WHERE symbol=? ORDER BY ts", st.db,
                             params=(symbol,)).set_index("ts")["long_pct"]
    if book.empty:
        df["long_pct"] = np.nan
    else:
        # the bar is decided at its close, so use the newest snapshot taken by then
        at_close = df.index + TF_MS[tf]
        merged = book.reindex(book.index.union(at_close)).ffill()
        df["long_pct"] = merged.reindex(at_close).to_numpy()
    df["usd_mom"] = _usd_momentum(st, tf, symbols, df.index)
    df["usd_side"] = float(usd_side(symbol))
    df["news"] = 0.0
    return df


def sync_all(st: Storage, symbols, tf: str, days: int, book_every: int) -> None:
    try:
        sync_financing(st, symbols)
    except Exception as e:
        log.warning("financing татаж чадсангүй: %s", e)
    for s in symbols:
        n = sync_candles(st, s, tf, days)
        b = sync_book(st, s, days, book_every)
        log.info("%s: +%d лаа, +%d position book", s, n, b)
