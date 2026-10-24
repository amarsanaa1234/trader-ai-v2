"""Commodities market plug (added 2026-10-19): metals, energy and grains on OANDA.

Same OANDA connection, data and broker as FX (markets/fx), but its own symbols,
evidence, risk settings, database and bot. Interest-rate carry/changes are left
out (a barrel of oil has no central bank); the dollar, bond yields, big funds'
positioning (COT), stocks, trend and sessions stay.

The futures curve (curve.py, 2026-10-20) replaces the interest-rate carry:
  * curve      - backwardation (stock short) favours longs, contango (glut) shorts
  * curve_chg  - the curve tightening / loosening over the last 4 weeks
  * swap_block - no new trade whose yearly swap would cost more than this
and the backtest charges each day's swap from the curve as it was then.
"""
import numpy as np
import pandas as pd

from markets.fx import data
from markets.fx.adapter import FX, _bars_per_day

from . import curve

KEEP = ("crowd", "usd", "tsmom", "cot", "xasset", "session", "orders", "yields", "month_end")
NEW = ("curve", "curve_chg")


class Commodities(FX):
    name = "commodities"
    label = "🥇 Түүхий эд"
    source_names = KEEP + NEW
    schema = data.SCHEMA + curve.SCHEMA
    default_params = {
        **{k: v for k, v in FX.default_params.items()
           if not k.startswith("w_") or k[2:] in KEEP or k == "w_news"},
        "w_curve": 0.0,       # new sources start with no say until a test shows they help
        "w_curve_chg": 0.0,
        "curve_scale": 0.2,   # 20% a year of carry = full signal
        # no trade whose swap costs > 10% a year (tools/curve_test.py 2026-10-20: walk-forward +0.45,
        # vault same, shared account +13.9%, random benchmark p=0.05); >= 0.95 = off
        "swap_block": 0.1,
    }
    param_space = {
        **{k: v for k, v in FX.param_space.items() if not k.startswith("w_") or k[2:] in KEEP},
        "w_curve": (-1.0, 1.5, False), "w_curve_chg": (-1.0, 1.5, False),
        "curve_scale": (0.05, 0.5, False), "swap_block": (0.05, 1.0, False),
        # FX narrowed these two on 2026-10-25; commodities keep their own range (owner asked FX only)
        "w_crowd": (-1.5, 1.0, False), "w_orders": (-1.0, 1.0, False),
    }
    rule_columns = ("long_pct", "usd_mom", "cot_net", "xasset", "session", "carry")
    rule_help = ("long_pct (% of OANDA clients long, 0-100; gold only), "
                 "usd_mom (1-day US-dollar basket log-move, + = dollar stronger, e.g. 0.003), "
                 "cot_net (speculators' net futures position, share of open interest, -0.5..0.5), "
                 "xasset (-1..1 pull from stocks), session (-1..1 London/New York range breakout), "
                 "carry (futures curve per year: + = backwardation/shortage, - = contango/glut, e.g. -0.3)")

    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        allp = super().sources(df, p)
        out = {k: allp[k] for k in KEEP}
        c = df["carry"] if "carry" in df else pd.Series(np.nan, index=df.index)
        scale = p.get("curve_scale", 0.2)
        out["curve"] = np.tanh(c / scale).fillna(0)
        out["curve_chg"] = np.tanh(c.diff(20 * _bars_per_day(df)) / scale).fillna(0)
        return out

    def side_filter(self, df: pd.DataFrame, p: dict):
        """(long_bad, short_bad): the swap for that side would cost more than swap_block a year."""
        lim = p.get("swap_block", 1.0)
        if lim >= 0.95 or "carry" not in df:
            return None
        c = df["carry"].to_numpy()
        return np.nan_to_num(c, nan=0.0) < -lim, np.nan_to_num(c, nan=0.0) > lim

    def daily_sync(self, st) -> None:
        super().daily_sync(st)
        curve.sync(st, self.symbols)

    def sync(self, st, symbols=None) -> None:
        super().sync(st, symbols)
        curve.sync(st, symbols or self.symbols)

    def load_frame(self, st, symbol: str, tf: str | None = None) -> pd.DataFrame:
        tf = tf or self.timeframe
        df = super().load_frame(st, symbol, tf)
        if df.empty:
            return df
        known = curve.load(st, symbol)
        carry = data._at_close(known, df.index, tf) if not known.empty else np.full(len(df), np.nan)
        # a value older than 10 days (the data stopped) is not used
        last = data._at_close(pd.Series(known.index, index=known.index), df.index, tf) \
            if not known.empty else np.full(len(df), np.nan)
        stale = ~(df.index.to_numpy() + data.TF_MS[tf] - np.nan_to_num(last, nan=-1e18) < 10 * curve.DAY_MS)
        carry = np.where(stale, np.nan, carry)
        df["carry"] = carry
        # swap as it was then: longs pay -carry a year (falls back to today's OANDA rate when unknown)
        row = st.db.execute("SELECT days FROM financing WHERE symbol=?", (symbol,)).fetchone()
        mult = data.rollover_days(row["days"] if row else None, df.index, tf)
        df["hold_cost"] = np.where(np.isnan(carry), df["hold_cost"], -carry / 365 * mult)
        return df


MARKET = Commodities()
