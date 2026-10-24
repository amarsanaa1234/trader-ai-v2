"""FX market data from OANDA: mid candles, OANDA clients' positioning (position
book), financing (swap) rates, prices of related markets (oil, gold, copper,
S&P 500) and the session ranges; plus the macro series from macro.py.

Every column is aligned to the moment the candle CLOSES (when the bot decides),
so nothing from the future leaks into a backtest."""
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from core.storage import Storage, now_ms

from . import macro
from .oanda import OandaError, data_client

log = logging.getLogger(__name__)
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
GRAN = {"5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D"}
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": DAY_MS}
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

SCHEMA = """
CREATE TABLE IF NOT EXISTS book (symbol TEXT, ts INTEGER, long_pct REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS orders (symbol TEXT, ts INTEGER, imbalance REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS financing (
    symbol TEXT PRIMARY KEY, long_rate REAL, short_rate REAL, days TEXT, updated INTEGER);
""" + macro.SCHEMA

# How each currency tends to move with other markets (sign, asset):
# commodity currencies follow their exports, safe havens rise when stocks fall.
CROSS_ASSET = {
    "AUD": [(0.5, "XCU_USD"), (0.5, "XAU_USD")],
    "NZD": [(0.5, "SPX500_USD"), (0.5, "XCU_USD")],
    "CAD": [(1.0, "BCO_USD")],
    "JPY": [(-1.0, "SPX500_USD")],
    "CHF": [(-0.5, "SPX500_USD"), (0.5, "XAU_USD")],
    # commodities: growth metals and oil follow stocks, gold is a safe haven
    "XAU": [(-0.5, "SPX500_USD")],
    "XAG": [(0.5, "SPX500_USD")],
    "XPT": [(0.5, "SPX500_USD")],
    "XCU": [(1.0, "SPX500_USD")],
    "BCO": [(0.5, "SPX500_USD")],
}
ASSET_SCALE = {"BCO_USD": 0.02, "XAU_USD": 0.01, "XCU_USD": 0.015, "SPX500_USD": 0.01}
# Government bond prices OANDA quotes daily (price up = yield down = currency usually weaker)
BONDS = {"USD": "USB02Y_USD", "EUR": "DE10YB_EUR", "GBP": "UK10YB_GBP"}


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
        # OANDA aligns 4h/daily candles to 17:00 New York by default (21:00/22:00 UTC); the bot
        # checks for closed candles on the UTC grid (00, 04, 08 ...), so ask for UTC-aligned
        # candles - otherwise every decision came up to 3 hours late.
        js = c.get(f"/instruments/{symbol}/candles", granularity=GRAN[tf], price="M",
                   dailyAlignment=0, alignmentTimezone="UTC",
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


def _position_value(b: dict) -> float | None:
    """% of OANDA clients' open positions that are long."""
    long_pct = sum(float(x["longCountPercent"]) for x in b["buckets"])
    short_pct = sum(float(x["shortCountPercent"]) for x in b["buckets"])
    return 100 * long_pct / (long_pct + short_pct) if long_pct + short_pct > 0 else None


def _order_value(b: dict, band: float = 0.01) -> float | None:
    """Waiting orders near the price: buy orders below it vs sell orders above it,
    in -1..1 (+ = more buyers waiting underneath than sellers overhead)."""
    px = float(b["price"])
    buy = sum(float(x["longCountPercent"]) for x in b["buckets"] if px * (1 - band) <= float(x["price"]) < px)
    sell = sum(float(x["shortCountPercent"]) for x in b["buckets"] if px < float(x["price"]) <= px * (1 + band))
    return (buy - sell) / (buy + sell) if buy + sell > 0 else None


BOOKS = {"book": ("positionBook", _position_value), "orders": ("orderBook", _order_value)}


def _book_at(symbol: str, ts_s: int | None, kind: str = "book"):
    endpoint, value = BOOKS[kind]
    params = {"time": str(ts_s)} if ts_s else {}
    try:
        b = data_client().get(f"/instruments/{symbol}/{endpoint}", **params)[endpoint]
    except OandaError as e:
        log.debug("%s %s %s: %s", symbol, endpoint, ts_s, e)
        return None
    try:
        v = value(b)
    except (ValueError, KeyError, TypeError):   # a few old snapshots have empty fields
        return None
    if v is None:
        return None
    ts = b.get("unixTime") or pd.Timestamp(b["time"]).timestamp()   # the books ignore the UNIX header
    return int(float(ts) * 1000), v


def sync_book(st: Storage, symbol: str, days: int, every_hours: int = 6, kind: str = "book") -> int:
    """OANDA keeps position- and order-book snapshots every 20 minutes back for years;
    history is filled at `every_hours` spacing, then the newest snapshot is added hourly.
    kind: "book" (clients' positions) or "orders" (clients' waiting orders)."""
    step = every_hours * HOUR_MS
    now = _book_at(symbol, None, kind)
    if now is None:          # OANDA keeps no book for this instrument (e.g. oil, gas): skip its history
        return 0
    last = st.db.execute(f"SELECT MAX(ts) m FROM {kind} WHERE symbol=?", (symbol,)).fetchone()["m"]
    start = max(last + step if last else 0, now_ms() - days * DAY_MS)
    start = (start // step + 1) * step
    times = list(range(start, now_ms() - step, step))
    rows = []
    if times:
        with ThreadPoolExecutor(8) as pool:
            rows = [r for r in pool.map(lambda t: _book_at(symbol, t // 1000, kind), times) if r]
    rows.append(now)
    st.db.executemany(f"INSERT OR REPLACE INTO {kind} VALUES (?, ?, ?)", [(symbol, ts, p) for ts, p in rows])
    st.db.commit()
    return len(rows)


def sync_spreads(st: Storage, symbols, path) -> dict:
    """Typical spread of each pair (median of the last ~3 months of hourly bid/ask closes,
    leaving out the rollover hours and weekends) -> the backtest's per-pair costs."""
    import json
    c = data_client()
    out = {}
    for s in symbols:
        js = c.get(f"/instruments/{s}/candles", granularity="H1", price="BA", count=2000)
        sp = []
        for k in js["candles"]:
            t = pd.Timestamp(float(k["time"]), unit="s", tz="UTC")
            if t.weekday() >= 5 or t.hour in (20, 21, 22):
                continue
            bid, ask = float(k["bid"]["c"]), float(k["ask"]["c"])
            sp.append((ask - bid) / ((ask + bid) / 2))
        if sp:
            out[s] = round(float(np.median(sp)), 7)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


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


# ---------- frame pieces ----------
def _closes(st: Storage, symbol: str, tf: str) -> pd.Series:
    return pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf=? ORDER BY ts", st.db,
                             params=(symbol, tf)).set_index("ts")["close"]


def _at_close(series: pd.Series, index: pd.Index, tf: str, step_ms: int | None = None) -> np.ndarray:
    """Value known when each candle of `index` closes. `series` is indexed by the time
    each value became known (default: candle open + its own length)."""
    known = series.copy()
    if step_ms is not None:
        known.index = known.index + step_ms
    at = pd.Index(index + TF_MS[tf])
    uniq = at.unique()   # the month-end lookup asks for the same moment many times
    known = known[~known.index.duplicated(keep="last")]
    return known.reindex(known.index.union(uniq)).ffill().reindex(uniq).reindex(at).to_numpy()


def _hold_cost(st: Storage, symbol: str, index: pd.Index, tf: str) -> np.ndarray:
    """Swap charged at the 17:00 New York rollover, as a symmetric long cost:
    long pays h, short pays -h, h = (shortRate - longRate) / 2 per year. (Both sides
    also pay OANDA's markup, about 1%/year on majors; small next to the spread.)"""
    row = st.db.execute("SELECT long_rate, short_rate, days FROM financing WHERE symbol=?", (symbol,)).fetchone()
    if not row:
        return np.zeros(len(index))
    h_day = (row["short_rate"] - row["long_rate"]) / 2 / 365
    return h_day * rollover_days(row["days"], index, tf)


def _hold_cost_then(st: Storage, symbol: str, index: pd.Index, tf: str, rate_diff: np.ndarray) -> np.ndarray:
    """Swap as it was at the time (audit 2026-10-25): today's OANDA swap moved by how much
    the two currencies' interest-rate gap has changed since. In 2020-21 US rates were ~0%,
    today ~3.6%; charging today's swap for all 8 years made old trades cost the wrong amount.
    The carry signal already used the historical rates, so cost and signal now agree."""
    row = st.db.execute("SELECT long_rate, short_rate, days FROM financing WHERE symbol=?", (symbol,)).fetchone()
    if not row:
        return np.zeros(len(index))
    h_now = (row["short_rate"] - row["long_rate"]) / 2           # long cost a year, today
    rd = pd.Series(rate_diff, dtype=float)
    known = rd.dropna()
    if known.empty:
        h_year = np.full(len(index), h_now)
    else:
        # a bigger base-minus-quote gap means the long earns more (pays less)
        h_year = (h_now - (rd - known.iloc[-1]) / 100).fillna(h_now).to_numpy()
    return h_year / 365 * rollover_days(row["days"], index, tf)


def rollover_days(days_csv: str | None, index: pd.Index, tf: str) -> np.ndarray:
    """Days of swap charged at the 17:00 New York rollover inside each candle (0 if none;
    3 on the day that covers the weekend). days_csv: Monday..Sunday from OANDA."""
    days = [int(x) for x in days_csv.split(",")] if days_csv else [1, 1, 3, 1, 1, 0, 0]
    opened = pd.to_datetime(index, unit="ms", utc=True).tz_convert("America/New_York")
    closed = opened + pd.Timedelta(milliseconds=TF_MS[tf])
    roll = closed.normalize() + pd.Timedelta(hours=17)
    charged = (opened < roll) & (roll <= closed)
    mult = np.array([days[d] for d in roll.weekday], dtype=float)
    return np.where(charged, mult, 0.0)


def _usd_momentum(st: Storage, tf: str, symbols, index: pd.Index) -> np.ndarray:
    """Broad US-dollar strength: average 1-day move of every USD pair, turned so +
    means USD got stronger."""
    bars = max(1, DAY_MS // TF_MS[tf])
    parts = []
    for s in symbols:
        side = usd_side(s)
        if side:
            parts.append(side * np.log(_closes(st, s, tf)).diff(bars))
    if not parts:
        return np.zeros(len(index))
    usd = pd.concat(parts, axis=1).mean(axis=1)
    return _at_close(usd, index, tf, TF_MS[tf])


def _cross_asset(st: Storage, symbol: str, tf: str, index: pd.Index) -> np.ndarray:
    """Pair score from related markets' last-day moves (e.g. oil up -> CAD up -> USD_CAD down)."""
    bars = max(1, DAY_MS // TF_MS[tf])
    moves = {}

    def move(asset):
        if asset not in moves:
            c = _closes(st, asset, tf)
            m = np.tanh(np.log(c).diff(bars) / ASSET_SCALE[asset]) if len(c) else pd.Series(dtype=float)
            moves[asset] = pd.Series(_at_close(m, index, tf, TF_MS[tf]), index=index).fillna(0.0)
        return moves[asset]

    def ccy(c):
        return sum((w * move(a) for w, a in CROSS_ASSET.get(c, [])), pd.Series(0.0, index=index))

    base, quote = symbol.split("_")
    return (ccy(base) - ccy(quote)).clip(-1, 1).to_numpy()


def _sessions(st: Storage, symbol: str, tf: str, index: pd.Index) -> np.ndarray:
    """Session breakout from 1h candles: after London opens (07:00 UTC) a close beyond
    the Tokyo range (00-07 UTC) = +1/-1; after New York opens (13:00 UTC) a close beyond
    the London morning range (07-13 UTC) adds another +1/-1. Averaged, 0 outside 07-21 UTC."""
    h = pd.read_sql_query("SELECT ts, high, low, close FROM candles WHERE symbol=? AND tf='1h' ORDER BY ts",
                          st.db, params=(symbol,)).set_index("ts")
    if h.empty:
        return np.zeros(len(index))
    dt = pd.to_datetime(h.index, unit="ms", utc=True)
    hour, day = dt.hour.to_numpy(), dt.normalize()

    def rng(lo_h, hi_h):
        m = (hour >= lo_h) & (hour < hi_h)
        g = h[m].groupby(day[m])
        return g["high"].max().reindex(day).to_numpy(), g["low"].min().reindex(day).to_numpy()

    tok_hi, tok_lo = rng(0, 7)
    lon_hi, lon_lo = rng(7, 13)
    c = h["close"].to_numpy()
    after_lon = (hour >= 7) & (hour < 21)
    after_ny = (hour >= 13) & (hour < 21)
    tok = np.where(after_lon & (c > tok_hi), 1.0, np.where(after_lon & (c < tok_lo), -1.0, 0.0))
    ny = np.where(after_ny & (c > lon_hi), 1.0, np.where(after_ny & (c < lon_lo), -1.0, 0.0))
    sess = pd.Series((tok + ny) / 2, index=h.index)
    # value of the last 1h candle that closed by the base candle's close
    return np.nan_to_num(_at_close(sess, index, tf, HOUR_MS))


def _yields(st: Storage, symbol: str, tf: str, index: pd.Index, days: int = 5) -> np.ndarray:
    """Daily bond-yield momentum, base minus quote: + = the base currency's yields rose
    more over the last week (bond prices fell more). Currencies without a bond: 0."""
    bars = max(1, days * DAY_MS // TF_MS[tf])

    def ylog(ccy):
        if ccy not in BONDS:
            return pd.Series(0.0, index=index)
        c = _closes(st, BONDS[ccy], tf)
        if c.empty:
            return pd.Series(0.0, index=index)
        return pd.Series(_at_close(-np.log(c).diff(bars), index, tf, TF_MS[tf]), index=index).fillna(0.0)

    base, quote = symbol.split("_")
    return (ylog(base) - ylog(quote)).to_numpy()


def _month_end(st: Storage, symbol: str, tf: str, index: pd.Index) -> np.ndarray:
    """Month-end rebalancing: in the last two business days of a month, funds whose US
    stocks rose sell dollars to rebalance (and buy them after a fall). Value: the expected
    push on this pair, +1/-1, 0 outside month-end or for pairs without USD."""
    side = usd_side(symbol)
    if not side:
        return np.zeros(len(index))
    spx = _closes(st, "SPX500_USD", tf)
    if spx.empty:
        return np.zeros(len(index))
    close_t = pd.to_datetime(index + TF_MS[tf], unit="ms", utc=True)
    spx_now = pd.Series(_at_close(spx, index, tf, TF_MS[tf]), index=index)
    month_start = close_t.tz_localize(None).to_period("M").to_timestamp().tz_localize("UTC")
    ms_idx = pd.Index(month_start.as_unit("ms").asi8)   # epoch ms whatever pandas' internal unit is
    spx_ms = pd.Series(_at_close(spx, ms_idx - TF_MS[tf], tf, TF_MS[tf]), index=index)
    mtd = spx_now / spx_ms - 1
    last_bd = (close_t + pd.offsets.BMonthEnd(0)).normalize()
    in_window = (close_t.normalize() >= last_bd - pd.offsets.BDay(1)) & (close_t.normalize() <= last_bd)
    return np.where(in_window, -side * np.sign(mtd.fillna(0).to_numpy()), 0.0)


def load_frame(st: Storage, symbol: str, tf: str, symbols) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND tf=? ORDER BY ts",
        st.db, params=(symbol, tf)).set_index("ts")
    if df.empty:
        return df
    at = df.index + TF_MS[tf]
    df["rate_diff"] = macro.rate_diff(st, symbol, at).to_numpy()
    df["hold_cost"] = _hold_cost_then(st, symbol, df.index, tf, df["rate_diff"].to_numpy())
    book = pd.read_sql_query("SELECT ts, long_pct FROM book WHERE symbol=? ORDER BY ts", st.db,
                             params=(symbol,)).set_index("ts")["long_pct"]
    df["long_pct"] = _at_close(book, df.index, tf) if not book.empty else np.nan
    df["usd_mom"] = _usd_momentum(st, tf, symbols, df.index)
    df["usd_side"] = float(usd_side(symbol))
    df["cot_net"] = macro.cot_net(st, symbol, at).to_numpy()
    df["xasset"] = _cross_asset(st, symbol, tf, df.index)
    df["session"] = _sessions(st, symbol, tf, df.index)
    orders = pd.read_sql_query("SELECT ts, imbalance FROM orders WHERE symbol=? ORDER BY ts", st.db,
                               params=(symbol,)).set_index("ts")["imbalance"]
    df["order_imb"] = _at_close(orders, df.index, tf) if not orders.empty else np.nan
    df["yields"] = _yields(st, symbol, tf, df.index)
    df["month_end"] = _month_end(st, symbol, tf, df.index)
    df["news"] = 0.0
    return df


def sync_all(st: Storage, symbols, tf: str, days: int, book_days: int, book_every: int, aux) -> None:
    for name, fn in (("financing", lambda: sync_financing(st, symbols)), ("macro", lambda: macro.sync_all(st))):
        try:
            fn()
        except Exception as e:
            log.warning("%s татаж чадсангүй: %s", name, e)
    tfs = [tf] + (["1h"] if tf != "1h" else [])     # 1h is always kept for the session ranges
    for s in list(symbols) + list(aux):
        n = {t: sync_candles(st, s, t, days) for t in tfs}
        b = sync_book(st, s, book_days, book_every) if s in symbols else 0
        o = sync_book(st, s, book_days, book_every, kind="orders") if s in symbols else 0
        log.info("%s: +%s лаа, +%d position book, +%d order book", s, n, b, o)
