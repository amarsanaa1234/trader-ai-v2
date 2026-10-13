"""FX market plug: OANDA v20 (practice account for demo)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from markets.base import Market

from . import data


class FX(Market):
    name = "fx"
    label = "💱 Валют"
    source_names = ("crowd", "usd")
    default_params = {
        # OANDA's own clients: unusually many of them long favours shorts (retail is
        # usually on the wrong side; the self-improver may flip this if data disagrees)
        "w_crowd": -0.4,
        # broad US-dollar trend across all pairs (momentum)
        "w_usd": 0.3,
        "usd_scale": 0.004,     # a 0.4% move of the whole dollar basket over 24h = full signal
    }
    param_space = {
        "w_crowd": (-1.5, 1.0, False), "w_usd": (-1.0, 1.5, False),
        "usd_scale": (0.002, 0.01, False),
    }
    rule_columns = ("long_pct", "usd_mom")
    rule_help = ("long_pct (% of OANDA clients long, 0-100, normal ~40-70), "
                 "usd_mom (24h US-dollar basket log-move, + = dollar stronger, e.g. 0.003)")
    schema = data.SCHEMA
    credentials = ("OANDA_TOKEN", "OANDA_ACCOUNT")

    # ---------- strategy plug ----------
    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        lp = df["long_pct"]
        mean, std = lp.rolling(720, min_periods=120).mean(), lp.rolling(720, min_periods=120).std()
        return {
            "crowd": ((lp - mean) / std).div(2).clip(-1, 1).fillna(0),
            "usd": (df["usd_side"] * np.tanh(df["usd_mom"] / p["usd_scale"])).fillna(0),
        }

    # ---------- data ----------
    @property
    def book_every(self) -> int:
        return int(self.cfg.get("book_every_hours", 6))

    def sync(self, st, symbols=None) -> None:
        data.sync_all(st, symbols or self.symbols, self.timeframe, self.history_days, self.book_every)

    def sync_symbol(self, st, symbol: str) -> None:
        data.sync_candles(st, symbol, self.timeframe, self.history_days)

    def daily_sync(self, st) -> None:
        data.sync_financing(st, self.symbols)

    def load_frame(self, st, symbol: str, tf: str | None = None) -> pd.DataFrame:
        return data.load_frame(st, symbol, tf or self.timeframe, self.symbols)

    def price(self, symbol: str) -> float:
        from .broker import bid_ask
        from .oanda import data_client
        return bid_ask(data_client(), [symbol])[symbol]["mid"]

    def prices(self, symbols) -> dict:
        from .broker import bid_ask
        from .oanda import data_client
        return {s: q["mid"] for s, q in bid_ask(data_client(), list(symbols)).items()}

    def is_open(self, ts_ms: int | None = None) -> bool:
        """FX trades Sunday ~21:00 UTC to Friday ~21:00 UTC."""
        t = datetime.fromtimestamp((ts_ms or datetime.now(timezone.utc).timestamp() * 1000) / 1000, timezone.utc)
        wd, h = t.weekday(), t.hour
        return not (wd == 5 or (wd == 4 and h >= 21) or (wd == 6 and h < 21))

    def hourly(self, trader, equity: float, day: dict) -> None:
        # newest crowd snapshot once an hour (the brain uses it at the next candle)
        for s in self.symbols:
            try:
                data.sync_book(trader.st, s, 2, 1)
            except Exception:
                pass

    # ---------- execution ----------
    def make_broker(self, st, account):
        from .broker import make_broker
        return make_broker(st, account)


MARKET = FX()
