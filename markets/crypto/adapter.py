"""Crypto market plug: Binance USDⓈ-M futures (+ spot for the funding carry)."""
import logging
import os

import numpy as np
import pandas as pd

from markets.base import Market

from . import data

log = logging.getLogger(__name__)


def _zscore(s: pd.Series, win: int = 168) -> pd.Series:
    """Compare with the past week so a market's 'normal' bias is ignored."""
    s = np.log(s)
    return ((s - s.rolling(win, min_periods=24).mean()) / s.rolling(win, min_periods=24).std()).div(2).clip(-1, 1)


class Crypto(Market):
    name = "crypto"
    label = "₿ Крипто"
    source_names = ("funding", "fng", "crowd", "top", "oi", "tsmom", "premium", "stocks", "usd", "funding_chg")
    default_params = {
        "w_news": 0.0,        # owner, 2026-10-19: news can't be backtested, so it gets no say for now
        # added 2026-10-19 with weight 0 (= no effect) until the self-improver proves them:
        "w_tsmom": 0.0,       # multi-week trend of the coin itself (Strategy 2's momentum part)
        "tsmom_days": 40,
        "w_premium": 0.0,     # Coinbase premium: US buyers paying more than Binance (Strategy 2's other part)
        "w_stocks": 0.0,      # S&P 500 over the last day
        "w_usd": 0.0,         # US dollar over the last day
        "w_funding_chg": 0.0, # funding rising/falling over the last day
        "w_funding": -0.5,    # negative = contrarian: crowded longs (high funding) favour shorts
        "w_fng": -0.3,        # negative = contrarian: extreme greed favours shorts
        # Crowd positioning (Binance keeps only 30 days, so these stay fixed until
        # the bot has collected enough history to tune them):
        "w_crowd": -0.4,      # unusually many long accounts favour shorts
        "w_top": 0.2,         # top traders' position ratio
        "w_oi": 0.3,          # open interest growing in the direction of price
        "funding_extreme": 0.0003,
    }
    param_space = {
        "w_funding": (-1.5, 1.0, False), "w_fng": (-1.5, 1.0, False),
        "funding_extreme": (0.0001, 0.001, False),
        "w_tsmom": (-1.0, 1.5, False), "tsmom_days": (10, 90, True), "w_premium": (-1.0, 1.5, False),
        "w_stocks": (-1.0, 1.5, False), "w_usd": (-1.5, 1.0, False), "w_funding_chg": (-1.5, 1.5, False),
    }
    rule_columns = ("funding", "fng")
    rule_help = "funding (8h funding rate, e.g. 0.0001), fng (Fear&Greed 0-100)"
    schema = data.SCHEMA
    credentials = ("BINANCE_KEY", "BINANCE_SECRET")

    def account_creds(self, user) -> dict | None:
        if user.legacy and self.mode != "signal":   # v1-style single-user .env
            return {"binance_key": self.api_key, "binance_secret": self.api_secret} if self.api_key else None
        return super().account_creds(user)

    # ---------- config ----------
    @property
    def mode(self) -> str:
        # v1 used BINANCE_MODE; keep reading it so an old .env still works
        return (self.env("mode") or os.getenv("BINANCE_MODE") or self.cfg.get("mode", "signal")).strip().lower()

    @property
    def api_key(self) -> str:
        return (self.env("binance_key") or os.getenv("BINANCE_API_KEY", "")).strip()

    @property
    def api_secret(self) -> str:
        return (self.env("binance_secret") or os.getenv("BINANCE_API_SECRET", "")).strip()

    @property
    def carry_cfg(self) -> dict:
        c = {"enabled": True, "symbols": [], "min_apr": 0.15, "exit_apr": 0.03, **self.cfg.get("carry", {})}
        c["symbols"] = [s for s in c["symbols"] if s not in self.symbols]  # never the coins we trade directionally
        return c

    # ---------- strategy plug ----------
    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        c = df["close"]
        oi_chg = df["oi_value"].pct_change(24, fill_method=None)
        step = int(np.min(np.diff(df.index.to_numpy()[:200]))) if len(df) > 2 else 3_600_000
        bpd = max(1, 86_400_000 // step)                      # candles per day
        logc = np.log(c)
        day_vol = logc.diff(bpd).rolling(60 * bpd, min_periods=20 * bpd).std()
        days = int(p["tsmom_days"])
        out = {
            "funding": (df["funding"] / p["funding_extreme"]).clip(-1, 1),
            "fng": ((df["fng"] - 50) / 50).clip(-1, 1),
            "crowd": _zscore(df["global_ls"]).fillna(0),
            "top": _zscore(df["top_ls"]).fillna(0),
            "oi": (np.tanh(oi_chg * 10) * np.sign(c.pct_change(24))).fillna(0),
            "tsmom": np.tanh(logc.diff(days * bpd) / (day_vol * np.sqrt(days))).fillna(0),
            "funding_chg": np.tanh(df["funding"].diff(bpd) / p["funding_extreme"]).fillna(0),
        }
        for name, col, scale in (("premium", "premium_z", 2.0), ("stocks", "spx_move", 0.01), ("usd", "usd_move", 0.004)):
            out[name] = (np.tanh(df[col] / scale) if col in df else pd.Series(0.0, index=df.index)).fillna(0)
        return out

    # ---------- data ----------
    def sync(self, st, symbols=None) -> None:
        data.sync_all(st, symbols or self.symbols, self.timeframe, self.history_days)

    def sync_symbol(self, st, symbol: str) -> None:
        # 1h is always kept fresh too: Strategy 2 and the Coinbase premium use hourly prices
        for tf in {self.timeframe, "1h"}:
            data.sync_candles(st, symbol, tf, self.history_days)
        data.sync_funding(st, symbol, self.history_days)

    def daily_sync(self, st) -> None:
        for s in self.symbols:
            data.sync_positioning(st, s)
        data.sync_fng(st)
        try:   # Coinbase prices for the premium source (the Strategy 2 service also keeps them fresh)
            from . import strategy2
            strategy2.sync_coinbase(st)
        except Exception as e:
            log.warning("Coinbase: %s", e)

    def load_frame(self, st, symbol: str, tf: str | None = None) -> pd.DataFrame:
        return data.load_frame(st, symbol, tf or self.timeframe)

    def price(self, symbol: str) -> float:
        return float(data.public_exchange().fetch_ticker(symbol)["last"])

    def prices(self, symbols) -> dict:
        tickers = data.public_exchange().fetch_tickers(list(symbols))
        return {s: t.get("last") for s, t in tickers.items()}

    # ---------- execution & extras ----------
    def make_broker(self, st, account):
        from .broker import make_broker
        return make_broker(st, account)

    def hourly(self, trader, equity: float, day: dict) -> None:
        if not self.carry_cfg["enabled"]:
            return
        from . import carry
        try:
            carry.manage(trader.st, trader.broker, equity, day["halted"], allow_new=not trader.paused(),
                         user=trader.acc.name)
        except Exception as e:
            log.warning("Carry (%s): %s", trader.acc.name, e)

    def close_carry(self, trader, carry_row: dict, reason: str) -> None:
        from . import carry
        carry.close(trader.st, trader.broker, carry_row, reason, user=trader.acc.name)

    def dashboard_extras(self, st) -> dict:
        from . import strategy2
        return {"brain2": strategy2.dashboard_panel(st)}

    def commands(self) -> dict:
        from . import cli
        return cli.COMMANDS


MARKET = Crypto()
