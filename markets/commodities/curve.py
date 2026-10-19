"""Futures curve (carry) of each commodity, from free data, owner request 2026-10-20.

OANDA's commodity CFDs trade at the cash (spot) price, while Yahoo's continuous
futures (GC=F, BZ=F ...) trade at the nearest contract's price. Their gap is the
front of the futures curve:

    carry = -ln(future / spot) * 365 / days_to_expiry      (per year)

  + (backwardation): stock is short right now, longs earn on the roll
  - (contango):      plenty of stock, holding a long costs money

It matches OANDA's own swap closely (2026-10-20: corn -16% vs OANDA -16%, wheat
-8% vs -9%, gas -64% vs -52%, Brent +33% vs +46%), because OANDA prices its swap
from the same curve. Yahoo keeps decades of the continuous series, so the 8-year
backtest gets the curve as it was at the time instead of today's swap.

OANDA's actual swap rates are also recorded every day (financing_hist) - the
paper account charges them and they let us compare later.
"""
import logging

import numpy as np
import pandas as pd
import requests

from core.storage import Storage, now_ms

log = logging.getLogger(__name__)
DAY_MS = 86_400_000
HOUR_MS = 3_600_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS curve (symbol TEXT, ts INTEGER, carry REAL, PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS financing_hist (
    symbol TEXT, ts INTEGER, long_rate REAL, short_rate REAL, PRIMARY KEY (symbol, ts));
"""

# Yahoo continuous future, price divisor (grains are quoted in cents), contract rule
YAHOO = {
    "XAU_USD": ("GC=F", 1, "metal", (2, 4, 6, 8, 10, 12)),
    "XAG_USD": ("SI=F", 1, "metal", (3, 5, 7, 9, 12)),
    "XPT_USD": ("PL=F", 1, "metal", (1, 4, 7, 10)),
    "XCU_USD": ("HG=F", 1, "metal", (3, 5, 7, 9, 12)),
    "BCO_USD": ("BZ=F", 1, "brent", None),
    "NATGAS_USD": ("NG=F", 1, "gas", None),
    "CORN_USD": ("ZC=F", 100, "grain", (3, 5, 7, 9, 12)),
    "WHEAT_USD": ("ZW=F", 100, "grain", (3, 5, 7, 9, 12)),
}
SETTLE_HOUR_UTC = 18      # futures settle ~18:15-19:30 UTC: compare with OANDA's 18:00-19:00 candle
KNOWN_HOUR_UTC = 21       # the value counts as known from 21:00 UTC that day
MIN_DAYS = 7              # closer to expiry the gap is too small to annualise reliably
SMOOTH_DAYS = 5           # median of 5 settlements (the two prices are ~1 hour apart)


def _bday(ts: pd.Timestamp, back: int = 0) -> pd.Timestamp:
    """Last business day of ts's month, `back` business days earlier."""
    return (ts + pd.offsets.BMonthEnd(0)).normalize() - pd.offsets.BDay(back)


def front_expiry(symbol: str, day: pd.Timestamp) -> pd.Timestamp:
    """Last trading day of the contract Yahoo's continuous future holds on `day`."""
    kind, months = YAHOO[symbol][2], YAHOO[symbol][3]
    day = day.normalize()
    if kind in ("brent", "gas"):       # a contract every month, expiring inside the current month
        back = 0 if kind == "brent" else 2      # Brent: last business day; gas: 3rd-last business day
        exp = _bday(day, back)
        if day > exp:
            exp = _bday(day + pd.offsets.MonthBegin(1), back)
        return exp
    # metals / grains: Yahoo moves to the next active month before first notice (end of the month before)
    for k in range(0, 15):
        m = (day + pd.offsets.MonthBegin(k) if k else day).to_period("M")
        if m.month not in months:
            continue
        month_start = m.to_timestamp()
        if day < _bday(month_start - pd.Timedelta(days=1)):
            if kind == "grain":                  # business day before the 15th
                return (month_start + pd.Timedelta(days=14) - pd.offsets.BDay(1)).normalize()
            return _bday(month_start, 2)        # metals: 3rd-last business day
    raise ValueError(f"no contract for {symbol} on {day}")


def yahoo_daily(ticker: str, days: int = 3300) -> pd.Series:
    """Settlement prices by exchange date for the last `days` days. (Yahoo turns a long
    `range` into monthly bars, so the period is asked for explicitly.)"""
    end = now_ms() // 1000
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"period1": end - days * 86_400, "period2": end, "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    days = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=days, dtype=float)
    return s[~s.index.duplicated(keep="last")].dropna()


def spot_at_settle(st: Storage, symbol: str) -> pd.Series:
    """OANDA's price at the futures settlement time, by UTC date."""
    h = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='1h' ORDER BY ts", st.db,
                          params=(symbol,)).set_index("ts")["close"]
    t = pd.to_datetime(h.index, unit="ms", utc=True)
    h = h[t.hour == SETTLE_HOUR_UTC]
    h.index = pd.to_datetime(h.index, unit="ms", utc=True).tz_localize(None).normalize()
    return h


def carry_series(symbol: str, fut: pd.Series, spot: pd.Series) -> pd.Series:
    """Annual carry per date (+ = backwardation), smoothed; NaN close to expiry."""
    div = YAHOO[symbol][1]
    both = pd.concat({"f": fut / div, "s": spot}, axis=1).dropna()
    if both.empty:
        return pd.Series(dtype=float)
    days = np.array([(front_expiry(symbol, d) - d).days for d in both.index], dtype=float)
    raw = -np.log(both["f"] / both["s"]) * 365 / np.where(days > 0, days, np.nan)
    raw[days < MIN_DAYS] = np.nan
    return raw.rolling(SMOOTH_DAYS, min_periods=2).median().clip(-3, 3)


def sync(st: Storage, symbols) -> int:
    """Curve history (9 years the first time, then the last 3 months) + today's OANDA swap."""
    added = 0
    for s in symbols:
        if s not in YAHOO:
            continue
        have = st.db.execute("SELECT COUNT(*) n FROM curve WHERE symbol=?", (s,)).fetchone()["n"]
        try:
            fut = yahoo_daily(YAHOO[s][0], 3300 if have < 1000 else 90)
        except Exception as e:
            log.warning("%s: Yahoo futures татаж чадсангүй: %s", s, e)
            continue
        c = carry_series(s, fut, spot_at_settle(st, s)).dropna()
        known = (c.index + pd.Timedelta(hours=KNOWN_HOUR_UTC)).tz_localize("UTC").as_unit("ms").asi8
        rows = [(s, int(t), float(v)) for t, v in zip(known, c.to_numpy())]
        st.db.executemany("INSERT OR REPLACE INTO curve VALUES (?, ?, ?)", rows)
        added += len(rows)
    today = now_ms() // DAY_MS * DAY_MS
    st.db.execute("INSERT OR REPLACE INTO financing_hist SELECT symbol, ?, long_rate, short_rate FROM financing",
                  (today,))
    st.db.commit()
    return added


def load(st: Storage, symbol: str) -> pd.Series:
    """Carry indexed by the time it became known (ms)."""
    return pd.read_sql_query("SELECT ts, carry FROM curve WHERE symbol=? ORDER BY ts", st.db,
                             params=(symbol,)).set_index("ts")["carry"]
