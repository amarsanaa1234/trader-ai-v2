"""Strategy 2 (SHADOW / paper only): daily momentum + Coinbase premium.

Runs next to the main bot on simulated money. It never places orders and
never touches the Binance account; it only keeps its own paper equity so the
two strategies can be compared on the same live candles.

  A. Daily time-series momentum (all coins): up over the last 40 days -> long,
     down -> short. Decided once a day, after the UTC day closes.
  F. Coinbase premium (BTC, ETH): Coinbase spot vs Binance futures price gap,
     as a z-score over 168 hours. z > 1 -> long, z < -1 -> short, otherwise
     keep the previous side.

Each coin's exposure is scaled to a 20%/yr volatility target (capped at 2x),
half the capital to A and half to F, so total exposure stays <= 2x equity.
Costs: 0.07% of every exposure change (fees + slippage) and funding.
Backtest of exactly this (research 2026-10-12): vault +8.3%, Sharpe 2.0.
"""
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from core.context import mkt
from core.settings import base_asset
from core.storage import Storage

log = logging.getLogger(__name__)

MOM_DAYS = 40
PREM_WINDOW, PREM_Z = 168, 1.0
TARGET_VOL, MAX_SCALE, VOL_WINDOW = 0.20, 2.0, 720
SHARE_A, SHARE_F = 0.5, 0.5          # capital split between the two parts
COST = 0.0007
START_EQUITY = 5000.0
PREMIUM_COINS = ("BTC", "ETH")
MODE = "s2"                          # its own label in equity_log
HOURS_YEAR = 8760


# ---------------- data ----------------
def _cb_table(st: Storage):
    st.db.execute("CREATE TABLE IF NOT EXISTS coinbase_candles "
                  "(product TEXT, ts INTEGER, close REAL, volume REAL, PRIMARY KEY(product, ts))")


def sync_coinbase(st: Storage, days_back: int = 730) -> int:
    """Hourly Coinbase BTC-USD / ETH-USD closes (public API, 300 candles per call)."""
    _cb_table(st)
    added = 0
    end = int(time.time()) // 3600 * 3600
    for coin in PREMIUM_COINS:
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


def load(st: Storage):
    """Aligned hourly closes, funding charges and Coinbase closes (index = candle open ms)."""
    _cb_table(st)
    frames = {base_asset(s): mkt().load_frame(st, s, "1h") for s in mkt().symbols}
    idx = pd.Index(sorted(set.intersection(*(set(f.index) for f in frames.values()))))
    close = pd.DataFrame({c: f["close"].reindex(idx) for c, f in frames.items()})
    funding = pd.DataFrame({c: f["hold_cost"].reindex(idx).fillna(0) for c, f in frames.items()})
    cb = pd.DataFrame(index=idx)
    for coin in PREMIUM_COINS:
        q = pd.read_sql_query("SELECT ts, close FROM coinbase_candles WHERE product=? ORDER BY ts", st.db,
                              params=(f"{coin}-USD",)).set_index("ts")["close"]
        cb[coin] = q.reindex(idx)
    return close, funding, cb


# ---------------- signals ----------------
def components(close: pd.DataFrame, cb: pd.DataFrame):
    """(momentum side, premium side, vol scale) per coin per hour; a value at
    hour t is decided with data up to t's close and held during t+1."""
    idx = close.index
    dt = pd.to_datetime(idx, unit="ms", utc=True)
    rets = close.pct_change()
    vol = rets.rolling(VOL_WINDOW, min_periods=200).std() * np.sqrt(HOURS_YEAR)
    scale = (TARGET_VOL / vol).clip(upper=MAX_SCALE).fillna(0)

    daily = close.set_axis(dt).resample("1D").last()
    mom_d = np.sign(daily / daily.shift(MOM_DAYS) - 1)
    mom_d.index = mom_d.index + pd.Timedelta(hours=23)       # known once the day's last candle has closed
    mom = mom_d.reindex(dt, method="ffill").set_axis(idx).fillna(0)

    prem = pd.DataFrame(0.0, index=idx, columns=close.columns)
    for coin in PREMIUM_COINS:
        if coin not in close or coin not in cb:
            continue
        p = cb[coin] / close[coin] - 1
        z = ((p - p.rolling(PREM_WINDOW).mean()) / p.rolling(PREM_WINDOW).std()).to_numpy()
        side, out = 0, np.zeros(len(z))
        for i, zi in enumerate(z):       # sticky: keep the side until the opposite extreme
            if not np.isnan(zi):
                side = 1 if zi > PREM_Z else (-1 if zi < -PREM_Z else side)
            out[i] = side
        prem[coin] = out
    return mom, prem, scale


def weights(close: pd.DataFrame, cb: pd.DataFrame) -> pd.DataFrame:
    """Exposure per coin as a fraction of equity (+ long, - short)."""
    mom, prem, scale = components(close, cb)
    n = len(close.columns)
    w = SHARE_A / n * mom * scale
    for coin in PREMIUM_COINS:
        if coin in w:
            w[coin] += SHARE_F / len(PREMIUM_COINS) * prem[coin] * scale[coin]
    return w


def backtest(close, funding, cb) -> pd.Series:
    """Hourly returns of the paper portfolio (same maths as the live step)."""
    w = weights(close, cb)
    held = w.shift(1).fillna(0)
    r = close.pct_change().fillna(0)
    cost = COST * w.diff().abs().fillna(w.abs()).sum(axis=1).shift(1).fillna(0)
    return (held * r).sum(axis=1) - (held * funding).sum(axis=1) - cost


# ---------------- live paper step ----------------
def step(st: Storage) -> dict | None:
    """Advance the paper portfolio over every newly closed hourly candle."""
    try:
        sync_coinbase(st)
    except Exception as e:  # premium part just holds its side if Coinbase is unreachable
        log.warning("Strategy 2: Coinbase sync failed: %s", e)
    close, funding, cb = load(st)
    if len(close) < VOL_WINDOW + MOM_DAYS * 24:
        return None
    w = weights(close, cb)
    rets = close.pct_change().fillna(0)
    s = st.get("s2")
    last = int(close.index[-1])
    if not s:
        w0 = w.iloc[-1]
        eq = START_EQUITY * (1 - COST * float(w0.abs().sum()))
        s = {"start_ts": last, "start_equity": START_EQUITY, "equity": eq, "last_ts": last,
             "weights": {c: float(v) for c, v in w0.items()}}
        st.set("s2", s)
        st.log_equity(MODE, eq, 0.0)
        log.info("Strategy 2 shadow started at %.2f", eq)
        return s
    new = [t for t in close.index if t > s["last_ts"]]
    if not new:
        return s
    eq, prev = s["equity"], pd.Series(s["weights"]).reindex(close.columns).fillna(0.0)
    for t in new:
        ret = float((prev * rets.loc[t]).sum() - (prev * funding.loc[t]).sum())
        nw = w.loc[t]
        cost = COST * float((nw - prev).abs().sum())
        eq *= 1 + ret - cost
        prev = nw
    s.update(equity=eq, last_ts=int(new[-1]), weights={c: float(v) for c, v in prev.items()})
    st.set("s2", s)
    st.log_equity(MODE, eq, 0.0)
    return s


def snapshot(st: Storage) -> dict | None:
    """For the dashboard / Telegram: equity, return and what each part says per coin."""
    s = st.get("s2")
    if not s:
        return None
    try:
        close, _, cb = load(st)
        mom, prem, _ = components(close, cb)
        parts = {c: {"momentum": int(mom[c].iloc[-1]), "premium": int(prem[c].iloc[-1]) if c in PREMIUM_COINS else None,
                     "weight": round(s["weights"].get(c, 0.0), 4)} for c in close.columns}
    except Exception as e:
        log.debug("snapshot: %s", e)
        parts = {c: {"momentum": None, "premium": None, "weight": round(v, 4)} for c, v in s["weights"].items()}
    return {"equity": s["equity"], "start_equity": s["start_equity"], "start_ts": s["start_ts"],
            "return": s["equity"] / s["start_equity"] - 1, "last_ts": s["last_ts"], "coins": parts,
            "exposure": round(sum(abs(v) for v in s["weights"].values()), 3)}


_panel_cache = {"ts": 0.0, "data": None}


def dashboard_panel(st: Storage) -> dict | None:
    """Strategy 2 (shadow) vs the main bot since Strategy 2 started. Cached 5 min
    because working out each coin's signals reads two years of candles."""
    if time.time() - _panel_cache["ts"] < 300:
        return _panel_cache["data"]
    out = None
    try:
        snap = snapshot(st)
        if snap:
            s2_hist = st.equity_history(MODE, snap["start_ts"])
            s1_hist = st.equity_history(mkt().mode, snap["start_ts"])
            step = max(1, len(s1_hist) // 400)
            s1_base = s1_hist[0]["equity"] if s1_hist else None
            out = {**snap,
                   "s1_return": (s1_hist[-1]["equity"] / s1_base - 1) if s1_base else None,
                   "s2_curve": [{"ts": h["ts"], "r": h["equity"] / snap["start_equity"] - 1} for h in s2_hist],
                   "s1_curve": [{"ts": h["ts"], "r": h["equity"] / s1_base - 1} for h in s1_hist[::step]] if s1_base else []}
    except Exception as e:
        log.debug("dashboard_panel: %s", e)
    _panel_cache.update(ts=time.time(), data=out)
    return out
