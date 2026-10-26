"""Commodities Strategy 2 (SHADOW / paper only): futures-curve carry ranking.

Every week the 8 commodities are ranked by the slope of their futures curve
(curve.py, per year): long the 3 in the steepest backwardation (shortage - longs
earn the roll / OANDA swap), short the 3 in the deepest contango (glut - shorts
earn it). Equal weights, held until the next week, exposure scaled to an 8%/year
volatility target from its own last 60 days.

It never places orders: it keeps its own paper equity next to the commodities bot
so the two can be compared on the same live prices.

Research (tools/carry_variants_test.py, 8 years, 2026-10-26): research years
+7.5%/yr Sharpe 0.90 max drawdown 12.6%, vault +11.3%/yr Sharpe 1.29; beat random
rankings (p=0.000 research, 0.023 vault); 7 of 9 years positive (2021 -6.5%).
The live swap OANDA charges differs from the curve on some (gas, Brent), so real
results may be a little lower.
"""
import logging
import time

import numpy as np
import pandas as pd

from core.context import mkt
from core.storage import Storage

from . import curve

log = logging.getLogger(__name__)

K, HOLD_DAYS = 3, 5
TARGET_VOL, VOL_DAYS = 0.08, 60
MAX_SCALE = 1.5                 # 6 legs x 1/3 = 2x gross -> at most 3x
START_EQUITY = 10_000.0
MODE = "s2"


def load(st: Storage, symbols):
    """Daily closes (last 4h close of each UTC weekday) and the curve carry known that day."""
    closes, carry = {}, {}
    for s in symbols:
        df = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='4h' ORDER BY ts", st.db,
                               params=(s,))
        c = df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))["close"]
        closes[s] = c.groupby(c.index.normalize()).last()
        k = curve.load(st, s)
        if not k.empty:
            k.index = pd.to_datetime(k.index, unit="ms", utc=True).normalize()
            carry[s] = k.groupby(k.index).last()
    px = pd.DataFrame(closes).dropna(how="all")
    px = px[px.index.day_of_week < 5].ffill()
    cy = pd.DataFrame(carry).reindex(px.index).ffill(limit=10)
    keep = cy.notna().sum(axis=1) >= len(symbols) - 2
    px, cy = px[keep], cy[keep]
    cy = cy.apply(lambda row: row.fillna(row.median()), axis=1)   # a missing curve ranks in the middle
    return px, cy


def weights(cy: pd.DataFrame) -> pd.DataFrame:
    w = np.zeros(cy.shape)
    last = np.zeros(cy.shape[1])
    vals = cy.to_numpy()
    for i in range(len(cy)):
        if i % HOLD_DAYS == 0 and np.isfinite(vals[i]).all():
            order = np.argsort(vals[i])
            last = np.zeros(cy.shape[1])
            last[order[-K:]] = 1.0 / K
            last[order[:K]] = -1.0 / K
        w[i] = last
    return pd.DataFrame(w, index=cy.index, columns=cy.columns)


def daily_returns(px, cy, w):
    """Price move + curve carry earned - spread on every weight change; volatility-targeted
    with past data only. Returns (net daily returns, scale)."""
    spread = pd.Series({s: 2 * mkt().costs_for(s)["maker"] for s in px.columns})
    ret = np.log(px).diff().fillna(0.0)
    wy = w.shift(1).fillna(0.0)
    raw = ((wy * ret).sum(axis=1) + (wy * cy).sum(axis=1) / 365 * (7 / 5)
           - (w.diff().abs().fillna(0.0) * spread).sum(axis=1))
    scale = (TARGET_VOL / (raw.rolling(VOL_DAYS, min_periods=20).std().shift(1) * np.sqrt(252))).clip(upper=MAX_SCALE)
    return (raw * scale).fillna(0.0), scale


def step(st: Storage) -> dict | None:
    """Advance the paper portfolio over every newly closed day (cheap; called hourly)."""
    px, cy = load(st, mkt().symbols)
    if len(px) < 200:
        return None
    w = weights(cy)
    rets, scale = daily_returns(px, cy, w)
    last = int(px.index[-1].value // 1_000_000)
    sc = float(scale.iloc[-1]) if np.isfinite(scale.iloc[-1]) else 0.0
    held = w.iloc[-1] * sc
    s = st.get(MODE)
    if not s:
        s = {"start_ts": last, "start_equity": START_EQUITY, "equity": START_EQUITY, "last_ts": last,
             "weights": {c: float(v) for c, v in held.items()}}
        st.set(MODE, s)
        st.log_equity(MODE, START_EQUITY, 0.0)
        log.info("Commodities Strategy 2 shadow started at %.2f", START_EQUITY)
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
    """Strategy 2 vs the commodities bot since Strategy 2 started (cached 5 min)."""
    if time.time() - _panel_cache["ts"] < 300:
        return _panel_cache["data"]
    out = None
    try:
        s = st.get(MODE)
        if s:
            _, cy = load(st, mkt().symbols)
            now = cy.iloc[-1] if len(cy) else pd.Series(dtype=float)
            s2_hist = st.equity_history(MODE, s["start_ts"])
            s1_hist = st.equity_history(mkt().mode, s["start_ts"])
            stp = max(1, len(s1_hist) // 400)
            s1_base = s1_hist[0]["equity"] if s1_hist else None
            out = {"kind": "carry", "metric_label": "Curve (жилд)",
                   "note": "долоо хоног бүр хамгийн их хомсдолтой (backwardation) 3 түүхий эдийг long, "
                           "хамгийн их илүүдэлтэй (contango) 3-ыг short барина (paper, захиалга тавихгүй)",
                   "equity": s["equity"], "start_equity": s["start_equity"], "start_ts": s["start_ts"],
                   "return": s["equity"] / s["start_equity"] - 1, "last_ts": s["last_ts"],
                   "currency": mkt().currency,
                   "exposure": round(sum(abs(v) for v in s["weights"].values()), 3), "fear": False,
                   "coins": {c: {"rate": None if pd.isna(now.get(c)) else round(float(now[c]) * 100, 1),
                                 "weight": round(s["weights"].get(c, 0.0), 4)} for c in mkt().symbols},
                   "s1_return": (s1_hist[-1]["equity"] / s1_base - 1) if s1_base else None,
                   "s2_curve": [{"ts": h["ts"], "r": h["equity"] / s["start_equity"] - 1} for h in s2_hist],
                   "s1_curve": [{"ts": h["ts"], "r": h["equity"] / s1_base - 1} for h in s1_hist[::stp]]
                   if s1_base else []}
    except Exception as e:
        log.debug("dashboard_panel: %s", e)
    _panel_cache.update(ts=time.time(), data=out)
    return out
