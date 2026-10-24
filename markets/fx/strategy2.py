"""FX Strategy 2 (SHADOW / paper only): currency carry ranking.

A portfolio strategy, not an entry signal. Every week the 8 currencies (USD EUR
JPY GBP AUD CAD CHF NZD) are ranked by their short-term interest rate; it holds
the 3 highest-yielding long and the 3 lowest short, equal weights, until the
next week. When fear jumps (VIX up more than 20% in 5 days) it steps aside,
because carry trades crash together in a panic. Exposure is scaled to an
8%/year volatility target from its own last 60 days.

It never places orders: it keeps its own paper equity next to the FX bot so the
two can be compared on the same live prices.

Research (tools/currency_rank_test.py, 8 years, 2026-10-22): the only FX method
that beat random rankings in both the research years (p=0.02) and the vault
(p=0.008); research +2.8%/yr Sharpe 0.33 max drawdown 11% with the VIX filter,
vault +11.1%/yr. Expect a small, steady edge, with losing years (2018).
"""
import logging
import time

import numpy as np
import pandas as pd

from core.context import mkt
from core.storage import Storage

log = logging.getLogger(__name__)

CCYS = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "NZD"]
VS_USD = {"EUR": ("EUR_USD", 1), "GBP": ("GBP_USD", 1), "AUD": ("AUD_USD", 1), "NZD": ("NZD_USD", 1),
          "JPY": ("USD_JPY", -1), "CAD": ("USD_CAD", -1), "CHF": ("USD_CHF", -1)}
K, HOLD_DAYS = 3, 5                     # long/short 3 currencies, re-ranked every 5 trading days
VIX_JUMP, VIX_DAYS = 1.2, 5             # flat after VIX rose > 20% in 5 days
TARGET_VOL, VOL_DAYS = 0.08, 60
MAX_SCALE = 1.5                         # 6 legs x 1/3 = 2x gross, so at most 3x: the FX bot's leverage cap
COST_PER_TURNOVER = 0.00015             # ~ a major's spread + slippage per unit of weight traded
MARKUP_PER_YEAR = 0.01                  # OANDA's financing markup on the gross position
YEAR = 252
START_EQUITY = 10_000.0
MODE = "s2"                             # its own label in equity_log


def _closes(st: Storage, symbol: str) -> pd.Series:
    df = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='4h' ORDER BY ts", st.db,
                           params=(symbol,))
    s = df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))["close"]
    return s[s.index.hour == 20]                     # the candle that closes at 00:00 UTC ~ New York close


def _asof(st: Storage, sql: str, params, at_ms: np.ndarray) -> np.ndarray:
    s = pd.read_sql_query(sql, st.db, params=params).drop_duplicates("ts").set_index("ts")["v"]
    if s.empty:
        return np.full(len(at_ms), np.nan)
    return s.reindex(s.index.union(at_ms)).ffill().reindex(at_ms).to_numpy()


def sync_vix(st: Storage) -> int:
    """VIX (US stock-market fear index) daily from FRED, public the next day. Fetched at most
    once every 12 hours."""
    st.db.execute("CREATE TABLE IF NOT EXISTS vix (ts INTEGER PRIMARY KEY, v REAL)")
    if time.time() - (st.get("s2_vix_synced") or 0) < 12 * 3600:
        return 0
    from .macro import _fred
    s = _fred("VIXCLS")
    rows = [(int((t + pd.Timedelta(days=1)).timestamp() * 1000), float(v)) for t, v in s.items()]
    st.db.executemany("INSERT OR REPLACE INTO vix VALUES (?, ?)", rows)
    st.db.commit()
    st.set("s2_vix_synced", time.time())
    return len(rows)


def load(st: Storage):
    """Daily log price of each currency in USD, its interest rate (as public then) and VIX."""
    st.db.execute("CREATE TABLE IF NOT EXISTS vix (ts INTEGER PRIMARY KEY, v REAL)")
    lp = pd.DataFrame({c: sgn * np.log(_closes(st, sym)) for c, (sym, sgn) in VS_USD.items()}).dropna()
    lp = lp[lp.index.dayofweek < 5]
    lp["USD"] = 0.0
    at = lp.index.as_unit("ms").asi8
    rt = pd.DataFrame({c: _asof(st, "SELECT ts, rate v FROM rates WHERE ccy=? ORDER BY ts", (c,), at) for c in CCYS},
                      index=lp.index)
    vix = pd.Series(_asof(st, "SELECT ts, v FROM vix ORDER BY ts", (), at), index=lp.index)
    keep = rt.notna().all(axis=1)
    return lp[CCYS][keep], rt[keep], vix[keep]


def weights(lp: pd.DataFrame, rt: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """Target weights per day (before volatility scaling); decided from data known that day."""
    w = pd.DataFrame(0.0, index=lp.index, columns=CCYS)
    last = pd.Series(0.0, index=CCYS)
    for i, t in enumerate(lp.index):
        if i % HOLD_DAYS == 0:
            order = list(rt.loc[t].sort_values().index)
            last = pd.Series(0.0, index=CCYS)
            last[order[-K:]] = 1.0 / K
            last[order[:K]] = -1.0 / K
        w.loc[t] = last
    fear = (vix / vix.shift(VIX_DAYS) > VIX_JUMP).shift(1, fill_value=False)   # yesterday's VIX
    return w.mul((~fear).astype(float), axis=0)


def daily_returns(lp, rt, w) -> pd.Series:
    """Net return of holding yesterday's weights over each day, volatility-targeted with past data only."""
    ret = lp.diff().fillna(0.0)
    wy = w.shift(1).fillna(0.0)
    raw = ((wy * ret).sum(axis=1) + (wy * rt / 100.0).sum(axis=1) / YEAR
           - wy.abs().sum(axis=1) / 2 * MARKUP_PER_YEAR / YEAR
           - w.diff().abs().sum(axis=1).fillna(0.0) * COST_PER_TURNOVER)
    vol = raw.rolling(VOL_DAYS, min_periods=20).std().shift(1) * np.sqrt(YEAR)
    return (raw * (TARGET_VOL / vol).clip(upper=MAX_SCALE)).fillna(0.0), (TARGET_VOL / vol).clip(upper=MAX_SCALE)


def step(st: Storage) -> dict | None:
    """Advance the paper portfolio over every newly closed day (cheap; called hourly)."""
    try:
        sync_vix(st)
    except Exception as e:      # without fresh VIX the fear filter just uses the last known value
        log.warning("FX Strategy 2: VIX sync failed: %s", e)
    lp, rt, vix = load(st)
    if len(lp) < 200:
        return None
    w = weights(lp, rt, vix)
    rets, scale = daily_returns(lp, rt, w)
    last = int(lp.index[-1].value // 1_000_000)
    held = (w.iloc[-1] * float(scale.iloc[-1] if np.isfinite(scale.iloc[-1]) else 0.0))
    s = st.get(MODE)
    if not s:
        s = {"start_ts": last, "start_equity": START_EQUITY, "equity": START_EQUITY, "last_ts": last,
             "weights": {c: float(v) for c, v in held.items()}}
        st.set(MODE, s)
        st.log_equity(MODE, START_EQUITY, 0.0)
        log.info("FX Strategy 2 shadow started at %.2f", START_EQUITY)
        return s
    new = rets[rets.index.as_unit("ms").asi8 > s["last_ts"]]
    if new.empty:
        return s
    eq = s["equity"] * float(np.prod(1 + new.to_numpy()))
    s.update(equity=eq, last_ts=last, weights={c: float(v) for c, v in held.items()})
    st.set(MODE, s)
    st.log_equity(MODE, eq, 0.0)
    return s


_panel_cache = {"ts": 0.0, "data": None}


def dashboard_panel(st: Storage) -> dict | None:
    """Strategy 2 vs the FX bot since Strategy 2 started (cached 5 min)."""
    if time.time() - _panel_cache["ts"] < 300:
        return _panel_cache["data"]
    out = None
    try:
        s = st.get(MODE)
        if s:
            _, rt, vix = load(st)
            rates = rt.iloc[-1] if len(rt) else pd.Series(dtype=float)
            s2_hist = st.equity_history(MODE, s["start_ts"])
            s1_hist = st.equity_history(mkt().mode, s["start_ts"])
            stp = max(1, len(s1_hist) // 400)
            s1_base = s1_hist[0]["equity"] if s1_hist else None
            out = {"kind": "carry", "equity": s["equity"], "start_equity": s["start_equity"], "start_ts": s["start_ts"],
                   "return": s["equity"] / s["start_equity"] - 1, "last_ts": s["last_ts"],
                   "currency": mkt().currency,
                   "exposure": round(sum(abs(v) for v in s["weights"].values()), 3),
                   "fear": bool(len(vix) > VIX_DAYS and vix.iloc[-1] / vix.iloc[-1 - VIX_DAYS] > VIX_JUMP),
                   "coins": {c: {"rate": None if pd.isna(rates.get(c)) else round(float(rates[c]), 2),
                                 "weight": round(s["weights"].get(c, 0.0), 4)} for c in CCYS},
                   "s1_return": (s1_hist[-1]["equity"] / s1_base - 1) if s1_base else None,
                   "s2_curve": [{"ts": h["ts"], "r": h["equity"] / s["start_equity"] - 1} for h in s2_hist],
                   "s1_curve": [{"ts": h["ts"], "r": h["equity"] / s1_base - 1} for h in s1_hist[::stp]]
                   if s1_base else []}
    except Exception as e:
        log.debug("dashboard_panel: %s", e)
    _panel_cache.update(ts=time.time(), data=out)
    return out
