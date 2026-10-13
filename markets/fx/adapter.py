"""FX market plug: OANDA v20 (practice account for demo)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from markets.base import Market

from . import data, macro

DAY_MS = 86_400_000


def _bars_per_day(df: pd.DataFrame) -> int:
    step = int(np.min(np.diff(df.index.to_numpy()[:200]))) if len(df) > 2 else 3_600_000
    return max(1, DAY_MS // step)


class FX(Market):
    name = "fx"
    label = "💱 Валют"
    source_names = ("crowd", "usd", "tsmom", "carry", "rates", "cot", "xasset", "session")
    default_params = {
        # OANDA's own clients: unusually many long favours shorts (retail is usually
        # on the wrong side; the self-improver may flip it if the data disagrees)
        "w_crowd": -0.4,
        "w_usd": 0.3,         # broad US-dollar trend across all pairs
        "usd_scale": 0.004,   # a 0.4% dollar-basket move in a day = full signal
        "w_tsmom": 0.5,       # time-series momentum: the pair's own multi-week trend
        "tsmom_days": 40,
        "w_carry": 0.3,       # interest-rate carry: hold the higher-yielding currency
        "w_rates": 0.4,       # rate differential moving in a currency's favour (3 months)
        "w_cot": -0.3,        # big speculators at an extreme -> contrarian
        "w_xasset": 0.3,      # oil -> CAD, gold/copper -> AUD, stocks down -> JPY/CHF up
        "w_session": 0.3,     # Tokyo-range breakout at the London open, London range at New York
    }
    param_space = {
        "w_crowd": (-1.5, 1.0, False), "w_usd": (-1.0, 1.5, False), "usd_scale": (0.002, 0.01, False),
        "w_tsmom": (-1.0, 1.5, False), "tsmom_days": (10, 120, True),
        "w_carry": (-1.0, 1.5, False), "w_rates": (-1.0, 1.5, False), "w_cot": (-1.5, 1.0, False),
        "w_xasset": (-1.0, 1.5, False), "w_session": (-1.0, 1.5, False),
    }
    rule_columns = ("long_pct", "usd_mom", "rate_diff", "cot_net", "xasset", "session")
    rule_help = ("long_pct (% of OANDA clients long, 0-100, normal ~40-70), "
                 "usd_mom (1-day US-dollar basket log-move, + = dollar stronger, e.g. 0.003), "
                 "rate_diff (base minus quote short-term interest rate in %, e.g. 0.7), "
                 "cot_net (speculators' net futures position base minus quote, share of open interest, -0.5..0.5), "
                 "xasset (-1..1 pull from oil/gold/copper/stocks), session (-1..1 London/New York range breakout)")
    schema = data.SCHEMA
    credentials = ("OANDA_TOKEN", "OANDA_ACCOUNT")

    # ---------- strategy plug ----------
    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        bpd = _bars_per_day(df)
        lp = df["long_pct"]
        mean, std = lp.rolling(30 * bpd, min_periods=5 * bpd).mean(), lp.rolling(30 * bpd, min_periods=5 * bpd).std()
        logc = np.log(df["close"])
        n = int(p["tsmom_days"]) * bpd
        day_vol = logc.diff(bpd).rolling(60 * bpd, min_periods=20 * bpd).std()
        tsmom = np.tanh(logc.diff(n) / (day_vol * np.sqrt(p["tsmom_days"])))
        rd = df["rate_diff"]
        cot = df["cot_net"]
        cot_z = (cot - cot.rolling(3 * 365 * bpd, min_periods=180 * bpd).mean()) / \
            cot.rolling(3 * 365 * bpd, min_periods=180 * bpd).std()
        return {
            "crowd": ((lp - mean) / std).div(2).clip(-1, 1).fillna(0),
            "usd": (df["usd_side"] * np.tanh(df["usd_mom"] / p["usd_scale"])).fillna(0),
            "tsmom": tsmom.fillna(0),
            "carry": np.tanh(rd / 2.0).fillna(0),
            "rates": np.tanh(rd.diff(90 * bpd) / 0.5).fillna(0),
            "cot": (cot_z / 2).clip(-1, 1).fillna(0),
            "xasset": df["xasset"].fillna(0),
            "session": df["session"].fillna(0),
        }

    def no_entry(self, index: pd.Index) -> np.ndarray:
        """Bars whose close falls in the daily rollover (wide spreads) or on Friday evening."""
        r = self.cfg.get("no_entry") or {}
        t = pd.to_datetime(np.asarray(index) + self.tf_ms(), unit="ms", utc=True)
        h = t.hour + t.minute / 60
        lo, hi = r.get("rollover_utc", [20.5, 22.5])
        block = (h >= lo) & (h <= hi)
        if r.get("friday_after_utc") is not None:
            block |= (t.weekday == 4) & (h >= r["friday_after_utc"])
        return np.asarray(block)

    def tags(self, df: pd.DataFrame) -> dict:
        """Situations the mistake memory may learn to avoid (side-independent)."""
        t = pd.to_datetime(df.index + self.tf_ms(), unit="ms", utc=True)
        h = t.hour
        return {"session_asia": pd.Series((h >= 0) & (h < 7), index=df.index),
                "session_london": pd.Series((h >= 7) & (h < 13), index=df.index),
                "session_newyork": pd.Series((h >= 13) & (h < 21), index=df.index),
                "session_late": pd.Series(h >= 21, index=df.index)}

    def spread_ok(self, symbol: str) -> tuple[bool, str]:
        from .broker import bid_ask
        from .oanda import data_client
        q = bid_ask(data_client(), [symbol])[symbol]
        spread = (q["ask"] - q["bid"]) / q["mid"]
        limit = float((self.cfg.get("risk") or {}).get("max_spread", 1.0))
        return spread <= limit, f"spread хэт өргөн ({spread:.3%} > {limit:.3%})"

    # ---------- data ----------
    @property
    def aux_symbols(self) -> list:
        return list(self.cfg.get("aux_symbols", []))

    def sync(self, st, symbols=None) -> None:
        data.sync_all(st, symbols or self.symbols, self.timeframe, self.history_days,
                      int(self.cfg.get("book_days", 730)), int(self.cfg.get("book_every_hours", 6)),
                      self.aux_symbols)

    def sync_symbol(self, st, symbol: str) -> None:
        for tf in {self.timeframe, "1h"}:
            data.sync_candles(st, symbol, tf, self.history_days)

    def daily_sync(self, st) -> None:
        data.sync_financing(st, self.symbols)
        macro.sync_all(st)

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
        # newest crowd snapshot and related-market candles once an hour
        for s in self.symbols:
            try:
                data.sync_book(trader.st, s, 2, 1)
            except Exception:
                pass
        for s in self.aux_symbols:
            try:
                data.sync_candles(trader.st, s, self.timeframe, self.history_days)
            except Exception:
                pass

    # ---------- execution ----------
    def make_broker(self, st, account):
        from .broker import make_broker
        return make_broker(st, account)


MARKET = FX()
