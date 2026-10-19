"""The plug every market implements.

The core (scoring, risk guard, backtest, walk-forward, shadow, dopamine,
Telegram, dashboard) is identical for all markets. A market only answers:
what do we trade, when is it open, where do candles come from, what extra
evidence does it have, what does trading cost, and how are orders placed.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.settings import DATA_ROOT

_UNITS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}


def tf_to_ms(tf: str) -> int:
    m = re.fullmatch(r"(\d+)([mhdw])", tf.strip().lower())
    if not m:
        raise ValueError(f"bad timeframe {tf!r}")
    return int(m.group(1)) * _UNITS[m.group(2)]


class Market:
    name = "base"
    label = "Base"
    # extra evidence this market adds to the score: name -> weight param "w_<name>"
    source_names: tuple = ()
    default_params: dict = {}
    param_space: dict = {}
    # raw data columns strategy-library rules may use, and how to describe them to the AI
    rule_columns: tuple = ()
    rule_help: str = ""
    schema: str = ""          # extra SQL tables this market stores
    credentials: tuple = ()   # per-user keys this market needs, read as USER<n>_<MARKET>_<KEY>

    def __init__(self):
        cfg_path = Path(__file__).resolve().parent / self.name / "config.yaml"
        self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.data_dir = DATA_ROOT / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ---------- config (env overrides yaml: <NAME>_<KEY>) ----------
    def env(self, key: str, default=None):
        return os.getenv(f"{self.name.upper()}_{key.upper()}", default)

    @property
    def mode(self) -> str:
        return (self.env("mode") or self.cfg.get("mode", "signal")).strip().lower()

    @property
    def symbols(self) -> list[str]:
        raw = self.env("symbols")
        return [s.strip() for s in raw.split(",") if s.strip()] if raw else list(self.cfg.get("symbols", []))

    @property
    def currency(self) -> str:
        """What account money is counted in (shown in messages and on the dashboard)."""
        return self.cfg.get("currency", "USDT")

    @property
    def timeframe(self) -> str:
        return self.env("timeframe") or self.cfg.get("timeframe", "1h")

    @property
    def history_days(self) -> int:
        return int(self.cfg.get("history_days", 730))

    @property
    def entry_order(self) -> str:
        return (self.env("entry_order") or self.cfg.get("entry_order", "maker")).lower()

    @property
    def costs(self) -> dict:
        """maker / taker fee and market-order slippage, as fractions of price."""
        return {"maker": 0.0002, "taker": 0.0005, "slippage": 0.0002, **self.cfg.get("costs", {})}

    def costs_for(self, symbol: str) -> dict:
        """Costs of one instrument (a market with very different spreads overrides this)."""
        return self.costs

    def force_exit(self, index: pd.Index, p: dict):
        """Bars at whose close an open position must be closed (e.g. before the FX weekend)."""
        return np.zeros(len(index), dtype=bool)

    def entry_filter(self, df: pd.DataFrame, p: dict):
        """Tunable bars-to-skip filter (e.g. FX: the day's range is already used up), or None."""
        return None

    def side_filter(self, df: pd.DataFrame, p: dict):
        """(long_bad, short_bad) bool arrays: bars where only that side may not start
        (e.g. commodities: holding it would cost too much swap), or None."""
        return None

    def tf_ms(self, tf: str | None = None) -> int:
        return tf_to_ms(tf or self.timeframe)

    # ---------- paths ----------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "bot.db"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "bot.log"

    @property
    def params_path(self) -> Path:
        return self.data_dir / "strategy_params.json"

    @property
    def params_history_dir(self) -> Path:
        return self.data_dir / "params_history"

    @property
    def library_path(self) -> Path:
        return self.data_dir / "strategy_library.json"

    # ---------- people ----------
    def account_creds(self, user) -> dict | None:
        """The user's keys for this market, {} in paper ("signal") mode, None if missing."""
        if self.mode == "signal":
            return {}
        vals = {k.lower(): user.cred(self.name, k) for k in self.credentials}
        return vals if all(vals.values()) else None

    def traders(self, users) -> list:
        """Users who trade this market right now (asked for it and have the keys)."""
        return [u for u in users if self.name in u.markets and self.account_creds(u) is not None]

    # ---------- strategy plug ----------
    def sources(self, df: pd.DataFrame, p: dict) -> dict:
        """Market-specific evidence in [-1, 1], keyed by source_names."""
        return {}

    def no_entry(self, index: pd.Index):
        """Bars on which no new trade may start (e.g. FX rollover); backtest and live alike."""
        return np.zeros(len(index), dtype=bool)

    def tags(self, df: pd.DataFrame) -> dict:
        """Extra side-independent situations (name -> bool Series) for the mistake memory."""
        return {}

    def spread_ok(self, symbol: str) -> tuple[bool, str]:
        """Live check right before an entry (e.g. spread unusually wide)."""
        return True, ""

    # ---------- data ----------
    def sync(self, st, symbols=None) -> None:
        raise NotImplementedError

    def sync_symbol(self, st, symbol: str) -> None:
        """Quick refresh before a decision (new candles and anything the score needs)."""
        raise NotImplementedError

    def daily_sync(self, st) -> None:
        pass

    def load_frame(self, st, symbol: str, tf: str | None = None) -> pd.DataFrame:
        """Candles indexed by open time (ms) with open/high/low/close/volume, hold_cost
        (cost rate charged on that candle for holding a long; negative = paid to longs)
        and any raw columns sources() needs."""
        raise NotImplementedError

    def price(self, symbol: str) -> float:
        raise NotImplementedError

    def prices(self, symbols) -> dict:
        return {s: self.price(s) for s in symbols}

    def is_open(self, ts_ms: int | None = None) -> bool:
        return True

    # ---------- execution & extras ----------
    def make_broker(self, st, account):
        """Broker for one person's account (account.user, account.creds, account.get/set state)."""
        raise NotImplementedError

    def hourly(self, trader, equity: float, day: dict) -> None:
        pass

    def close_carry(self, trader, carry_row: dict, reason: str) -> None:
        """Close a hedged carry position (only markets that run one)."""

    def dashboard_extras(self, st) -> dict:
        return {}

    def commands(self) -> dict:
        """Extra CLI commands: name -> (function(st), help text)."""
        return {}

    @property
    def news_feeds(self) -> dict:
        return dict(self.cfg.get("news_feeds", {}))

    @property
    def research_sources(self) -> dict:
        return dict(self.cfg.get("research_sources", {}))
