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
    source_names = ("crowd", "usd", "tsmom", "carry", "rates", "cot", "xasset", "session",
                    "orders", "yields", "month_end")
    default_params = {
        "w_news": 0.0,        # owner, 2026-10-19: news can't be backtested, so it gets no say for now
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
        "w_orders": 0.2,      # OANDA clients' waiting orders: buyers below vs sellers above
        "w_yields": 0.3,      # daily government-bond yield moves (US 2y, German/UK 10y)
        "w_month_end": 0.3,   # month-end rebalancing flows against the month's stock move
        "adr_block": 2.0,     # skip entries once the day's range > this x average day range (>=1.95 = off)
        "weekend_close": 0.0, # > 0.5: close open trades on Friday evening (avoid the weekend gap)
    }
    param_space = {
        # crowd and waiting orders only have OANDA history since 2024-10 (2 of the 8 test years),
        # so their weight is kept small until more history exists (audit 2026-10-25; was -1.5..1.0)
        "w_crowd": (-0.6, 0.6, False), "w_usd": (-1.0, 1.5, False), "usd_scale": (0.002, 0.01, False),
        "w_tsmom": (-1.0, 1.5, False), "tsmom_days": (10, 120, True),
        # COT contrarian only: going against extreme speculators paid in 10/10 years and 7/7
        # currencies (tools/cot_test.py, 2026-10-26); following them (live was +0.63) did not
        "w_carry": (-1.0, 1.5, False), "w_rates": (-1.0, 1.5, False), "w_cot": (-1.5, 0.0, False),
        "w_xasset": (-1.0, 1.5, False), "w_session": (-1.0, 1.5, False),
        "w_orders": (-0.5, 0.5, False), "w_yields": (-1.0, 1.5, False), "w_month_end": (-1.0, 1.0, False),
        "adr_block": (0.6, 2.0, False), "weekend_close": (0.0, 1.0, False),
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
            "orders": df["order_imb"].clip(-1, 1).fillna(0),
            "yields": np.tanh(df["yields"] / 0.005).fillna(0),
            "month_end": df["month_end"].fillna(0),
        }

    def entry_filter(self, df: pd.DataFrame, p: dict):
        """True = skip: today's high-low already exceeds adr_block x the average daily range
        of the last 20 days (the move is probably used up)."""
        if p.get("adr_block", 2.0) >= 1.95:
            return None
        day = pd.to_datetime(df.index, unit="ms", utc=True).normalize()
        g = pd.DataFrame({"h": df["high"].to_numpy(), "l": df["low"].to_numpy(), "d": day})
        today_range = g.groupby("d")["h"].cummax() - g.groupby("d")["l"].cummin()
        daily = (g.groupby("d")["h"].max() - g.groupby("d")["l"].min())
        adr = daily.rolling(20, min_periods=10).mean().shift(1).reindex(day).to_numpy()
        return np.asarray(today_range.to_numpy() > p["adr_block"] * adr)

    def force_exit(self, index, p: dict):
        """With weekend_close on: close at the last candle before the Friday close (21:00 UTC)."""
        if p.get("weekend_close", 0.0) <= 0.5:
            return np.zeros(len(index), dtype=bool)
        t = pd.to_datetime(np.asarray(index) + self.tf_ms(), unit="ms", utc=True)
        nxt = t + pd.Timedelta(milliseconds=self.tf_ms())
        return np.asarray((t.weekday == 4) & (t.hour < 21) & ((nxt.hour >= 21) | (nxt.weekday != 4)))

    def costs_for(self, symbol: str) -> dict:
        """Per-pair costs from the measured typical spread (spreads.json, made by sync);
        crosses like GBP_JPY cost 2-3x EUR_USD."""
        sp = self._spreads().get(symbol)
        through = float(self.cfg.get("fill_through_spread", 0.0))   # share of the spread a limit must be beaten by
        if not sp:
            c = dict(self.costs)
            c["fill_through"] = c["maker"] * 2 * through
            return c
        return {"maker": sp * 0.5, "taker": sp * 0.6, "slippage": sp * 0.15, "fill_through": sp * through}

    def _spreads(self) -> dict:
        import json
        path = self.data_dir / "spreads.json"
        if not path.exists():
            return {}
        mtime = path.stat().st_mtime
        if getattr(self, "_sp_cache", (None, None))[0] != mtime:
            self._sp_cache = (mtime, json.loads(path.read_text(encoding="utf-8")))
        return self._sp_cache[1]

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
                "session_late": pd.Series(h >= 21, index=df.index),
                "month_end": df["month_end"] != 0 if "month_end" in df else pd.Series(False, index=df.index)}

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
        data.sync_spreads(st, self.symbols, self.data_dir / "spreads.json")

    def sync_symbol(self, st, symbol: str) -> None:
        for tf in {self.timeframe, "1h"}:
            data.sync_candles(st, symbol, tf, self.history_days)

    def daily_sync(self, st) -> None:
        data.sync_financing(st, self.symbols)
        macro.sync_all(st)

    def load_frame(self, st, symbol: str, tf: str | None = None) -> pd.DataFrame:
        # the dollar basket: the traded pairs for FX; other OANDA markets list FX pairs in usd_basket
        return data.load_frame(st, symbol, tf or self.timeframe, self.cfg.get("usd_basket") or self.symbols)

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
            for kind in ("book", "orders"):
                try:
                    data.sync_book(trader.st, s, 2, 1, kind=kind)
                except Exception:
                    pass
        for s in self.aux_symbols:
            try:
                data.sync_candles(trader.st, s, self.timeframe, self.history_days)
            except Exception:
                pass
        if self.name == "fx":       # FX Strategy 2: paper carry ranking next to the bot (not commodities)
            from . import strategy2
            try:
                strategy2.step(trader.st)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("FX Strategy 2: %s", e)

    def dashboard_extras(self, st) -> dict:
        if self.name != "fx":
            return {}
        from . import strategy2
        return {"brain2": strategy2.dashboard_panel(st)}

    # ---------- execution ----------
    def make_broker(self, st, account):
        from .broker import make_broker
        return make_broker(st, account)


MARKET = FX()
