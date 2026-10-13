"""Slow macro data for FX, all free and with long history:

  * short-term interest rates per currency (FRED) -> carry and rate-change sources
  * CFTC Commitments of Traders: large speculators' net position in each
    currency future (weekly) -> COT source

Every value is stored with the time it became PUBLIC, so backtests never use
a number before it was released.
"""
import io
import logging

import pandas as pd
import requests

from core.storage import Storage, now_ms

log = logging.getLogger(__name__)
DAY_MS = 86_400_000

SCHEMA = """
CREATE TABLE IF NOT EXISTS rates (ccy TEXT, ts INTEGER, rate REAL, PRIMARY KEY (ccy, ts));
CREATE TABLE IF NOT EXISTS cot (ccy TEXT, ts INTEGER, net_pct REAL, PRIMARY KEY (ccy, ts));
"""

# (FRED series, monthly?) in order of preference. Monthly OECD 3-month rates are
# public about 6 weeks after the month; daily policy-rate series the next day.
RATE_SERIES = {
    "USD": [("DFF", False)],                    # Fed funds effective, daily
    "EUR": [("ECBDFR", False)],                 # ECB deposit facility, daily
    "GBP": [("IUDSOIA", False), ("IR3TIB01GBM156N", True)],   # SONIA, daily
    "JPY": [("IR3TIB01JPM156N", True)],
    "AUD": [("IR3TIB01AUM156N", True)],
    "CAD": [("IR3TIB01CAM156N", True)],
    "CHF": [("IR3TIB01CHM156N", True)],
    "NZD": [("IR3TIB01NZM156N", True)],
}
# CFTC contract codes (legacy futures-only report); USD has no matching future here
COT_CODES = {"EUR": "099741", "JPY": "097741", "GBP": "096742", "AUD": "232741",
             "CAD": "090741", "CHF": "092741", "NZD": "112741"}


def _fred(series: str) -> pd.Series:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    s = pd.to_numeric(df["value"], errors="coerce")
    s.index = pd.to_datetime(df["date"], utc=True)
    return s.dropna()


def sync_rates(st: Storage) -> int:
    rows = []
    for ccy, choices in RATE_SERIES.items():
        for series, monthly in choices:
            try:
                s = _fred(series)
            except Exception as e:
                log.warning("FRED %s: %s", series, e)
                continue
            if s.empty:
                continue
            # when it was public: monthly -> 6 weeks after the month ends; daily -> next day
            public = (s.index + pd.offsets.MonthEnd(1) + pd.Timedelta(days=45)) if monthly else \
                s.index + pd.Timedelta(days=1)
            rows += [(ccy, int(t.timestamp() * 1000), float(v)) for t, v in zip(public, s.to_numpy())]
            break
    st.db.execute("DELETE FROM rates")
    st.db.executemany("INSERT OR REPLACE INTO rates VALUES (?, ?, ?)", rows)
    st.db.commit()
    return len(rows)


def sync_cot(st: Storage, years: int = 10) -> int:
    since = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=years)
    rows = []
    for ccy, code in COT_CODES.items():
        try:
            r = requests.get("https://publicreporting.cftc.gov/resource/6dca-aqww.json", timeout=60, params={
                "cftc_contract_market_code": code, "$limit": 5000, "$order": "report_date_as_yyyy_mm_dd",
                "$where": f"report_date_as_yyyy_mm_dd >= '{since:%Y-%m-%d}'"})
            r.raise_for_status()
        except Exception as e:
            log.warning("CFTC %s: %s", ccy, e)
            continue
        for x in r.json():
            oi = float(x.get("open_interest_all") or 0)
            if oi <= 0:
                continue
            net = (float(x["noncomm_positions_long_all"]) - float(x["noncomm_positions_short_all"])) / oi
            # Tuesday's positions are published Friday 15:30 New York -> usable from Friday 21:00 UTC
            public = pd.Timestamp(x["report_date_as_yyyy_mm_dd"][:10], tz="UTC") + pd.Timedelta(days=3, hours=21)
            rows.append((ccy, int(public.timestamp() * 1000), net))
    st.db.executemany("INSERT OR REPLACE INTO cot VALUES (?, ?, ?)", rows)
    st.db.commit()
    return len(rows)


def _asof(st: Storage, table: str, col: str, ccy: str, at: pd.Index) -> pd.Series:
    """Latest public value at each time in `at` (ms)."""
    s = pd.read_sql_query(f"SELECT ts, {col} v FROM {table} WHERE ccy=? ORDER BY ts", st.db,
                          params=(ccy,)).drop_duplicates("ts").set_index("ts")["v"]
    if s.empty:
        return pd.Series(float("nan"), index=at)
    return s.reindex(s.index.union(at)).ffill().reindex(at)


def rate_diff(st: Storage, symbol: str, at: pd.Index) -> pd.Series:
    """Base minus quote short-term rate, in percent (e.g. AUD 4.5 - USD 3.8 = +0.7)."""
    base, quote = symbol.split("_")
    return _asof(st, "rates", "rate", base, at) - _asof(st, "rates", "rate", quote, at)


def cot_net(st: Storage, symbol: str, at: pd.Index) -> pd.Series:
    """Speculators' net position (share of open interest) in the base minus the quote
    currency future; USD counts as 0."""
    base, quote = symbol.split("_")
    b = _asof(st, "cot", "net_pct", base, at) if base in COT_CODES else pd.Series(0.0, index=at)
    q = _asof(st, "cot", "net_pct", quote, at) if quote in COT_CODES else pd.Series(0.0, index=at)
    return b - q


def sync_all(st: Storage) -> None:
    for name, fn in (("хүү", sync_rates), ("COT", sync_cot)):
        try:
            log.info("%s: %d мөр", name, fn(st))
        except Exception as e:   # one source failing must not stop the other
            log.warning("%s татаж чадсангүй: %s", name, e)


def last_update_age_days(st: Storage) -> float:
    row = st.db.execute("SELECT MAX(ts) m FROM cot").fetchone()
    return (now_ms() - (row["m"] or 0)) / DAY_MS
