"""Strategy library.

Strategies are plain JSON rules (no code), e.g.
  {"name": "RSI dip in uptrend",
   "long_entry":  [{"left": "rsi:14", "op": "<", "right": "30"}, {"left": "close", "op": ">", "right": "ema:200"}],
   "short_entry": [{"left": "rsi:14", "op": ">", "right": "70"}, {"left": "close", "op": "<", "right": "ema:200"}]}

They come from two places: classic textbook strategies seeded below, and
strategies the AI extracts from articles on the internet (research.py).
Every one is backtested on its own; only those that pass on unseen data are
"approved" and get a vote in the live decision.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import validation
from .indicators import atr, ema, rsi
from .context import mkt
from .strategy import default_params

log = logging.getLogger(__name__)
MIN_SCORE = 0.3

OPS = [">", "<", "crosses_above", "crosses_below"]
_CORE_COLUMNS = ("open", "high", "low", "close", "volume", "macd", "macd_signal", "macd_hist")


def feature_re() -> re.Pattern:
    """Allowed rule features: the core ones plus the active market's raw columns (crypto: funding, fng)."""
    cols = "|".join(_CORE_COLUMNS + tuple(mkt().rule_columns))
    return re.compile(rf"^({cols}"
                      r"|(ema|sma|rsi|atr|bb_upper|bb_lower|bb_mid|highest|lowest|volume_sma|roc):(\d{1,3}))$")


def feature_help() -> str:
    extra = f"{mkt().rule_help}, " if mkt().rule_help else ""
    return ("open, high, low, close, volume, macd, macd_signal, macd_hist, " + extra +
            "ema:N, sma:N, rsi:N, atr:N, bb_upper:N, bb_lower:N, bb_mid:N (Bollinger 2σ), "
            "highest:N / lowest:N (highest high / lowest low of the previous N bars), volume_sma:N, "
            f"roc:N (% change over N bars, e.g. 0.05 = +5%). N is 2-400 bars of {mkt().timeframe}.")


SEEDS = [
    {"name": "Golden / death cross", "source": "classic",
     "long_entry": [{"left": "ema:50", "op": "crosses_above", "right": "ema:200"}],
     "short_entry": [{"left": "ema:50", "op": "crosses_below", "right": "ema:200"}]},
    {"name": "RSI dip with the trend", "source": "classic",
     "long_entry": [{"left": "rsi:14", "op": "<", "right": "30"}, {"left": "close", "op": ">", "right": "ema:200"}],
     "short_entry": [{"left": "rsi:14", "op": ">", "right": "70"}, {"left": "close", "op": "<", "right": "ema:200"}]},
    {"name": "Bollinger breakout", "source": "classic",
     "long_entry": [{"left": "close", "op": "crosses_above", "right": "bb_upper:20"}],
     "short_entry": [{"left": "close", "op": "crosses_below", "right": "bb_lower:20"}]},
    {"name": "Bollinger mean reversion", "source": "classic",
     "long_entry": [{"left": "close", "op": "<", "right": "bb_lower:20"}, {"left": "rsi:14", "op": "<", "right": "35"}],
     "short_entry": [{"left": "close", "op": ">", "right": "bb_upper:20"}, {"left": "rsi:14", "op": ">", "right": "65"}]},
    {"name": "Turtle / Donchian 55", "source": "classic",
     "long_entry": [{"left": "close", "op": ">", "right": "highest:55"}],
     "short_entry": [{"left": "close", "op": "<", "right": "lowest:55"}]},
    {"name": "MACD cross with trend", "source": "classic",
     "long_entry": [{"left": "macd", "op": "crosses_above", "right": "macd_signal"},
                    {"left": "close", "op": ">", "right": "ema:200"}],
     "short_entry": [{"left": "macd", "op": "crosses_below", "right": "macd_signal"},
                     {"left": "close", "op": "<", "right": "ema:200"}]},
    {"name": "Volume breakout", "source": "classic",
     "long_entry": [{"left": "close", "op": ">", "right": "highest:24"},
                    {"left": "volume", "op": ">", "right": "volume_sma:24", "right_mult": 2}],
     "short_entry": [{"left": "close", "op": "<", "right": "lowest:24"},
                     {"left": "volume", "op": ">", "right": "volume_sma:24", "right_mult": 2}]},
    {"name": "Funding squeeze (contrarian)", "source": "classic",
     "long_entry": [{"left": "funding", "op": "<", "right": "-0.0003"}, {"left": "rsi:14", "op": "<", "right": "35"}],
     "short_entry": [{"left": "funding", "op": ">", "right": "0.0005"}, {"left": "rsi:14", "op": ">", "right": "65"}]},
    {"name": "Fear & Greed extremes", "source": "classic",
     "long_entry": [{"left": "fng", "op": "<", "right": "20"}, {"left": "close", "op": ">", "right": "ema:50"}],
     "short_entry": [{"left": "fng", "op": ">", "right": "80"}, {"left": "close", "op": "<", "right": "ema:50"}]},
    {"name": "EMA20 pullback in trend", "source": "classic",
     "long_entry": [{"left": "close", "op": ">", "right": "ema:200"},
                    {"left": "close", "op": "crosses_above", "right": "ema:20"}],
     "short_entry": [{"left": "close", "op": "<", "right": "ema:200"},
                     {"left": "close", "op": "crosses_below", "right": "ema:20"}]},
]


# ---------- rule evaluation ----------

def _feature(df: pd.DataFrame, name: str, cache: dict) -> pd.Series:
    if name in cache:
        return cache[name]
    m = feature_re().match(name)
    if not m:
        raise ValueError(f"unknown feature {name!r}")
    c = df["close"]
    if m.group(2):
        kind, n = m.group(2), max(2, min(400, int(m.group(3))))
        if kind == "ema":
            s = ema(c, n)
        elif kind == "sma":
            s = c.rolling(n).mean()
        elif kind == "rsi":
            s = rsi(c, n)
        elif kind == "atr":
            s = atr(df, n)
        elif kind in ("bb_upper", "bb_lower", "bb_mid"):
            mid, sd = c.rolling(n).mean(), c.rolling(n).std()
            s = {"bb_upper": mid + 2 * sd, "bb_lower": mid - 2 * sd, "bb_mid": mid}[kind]
        elif kind == "highest":
            s = df["high"].rolling(n).max().shift(1)
        elif kind == "lowest":
            s = df["low"].rolling(n).min().shift(1)
        elif kind == "volume_sma":
            s = df["volume"].rolling(n).mean()
        else:  # roc
            s = c.pct_change(n)
    else:
        kind = m.group(1)
        if kind.startswith("macd"):
            macd = ema(c, 12) - ema(c, 26)
            sig = ema(macd, 9)
            s = {"macd": macd, "macd_signal": sig, "macd_hist": macd - sig}[kind]
        else:
            s = df[kind]
    cache[name] = s
    return s


def _operand(df, value, cache):
    try:
        return float(value)
    except (TypeError, ValueError):
        return _feature(df, str(value).strip(), cache)


def _condition(df, cond, cache) -> pd.Series:
    left = _feature(df, cond["left"].strip(), cache)
    right = _operand(df, cond["right"], cache) * float(cond.get("right_mult") or 1.0)
    op = cond["op"]
    if op == ">":
        out = left > right
    elif op == "<":
        out = left < right
    else:
        diff = left - right
        out = (diff > 0) & (diff.shift(1) <= 0) if op == "crosses_above" else (diff < 0) & (diff.shift(1) >= 0)
    return out.fillna(False)


def validate_rules(strat: dict) -> str | None:
    """Returns an error message, or None if the rules are well-formed."""
    for side in ("long_entry", "short_entry"):
        conds = strat.get(side) or []
        if len(conds) > 5:
            return f"{side}: хэт олон нөхцөл"
        for c in conds:
            if c.get("op") not in OPS:
                return f"буруу op {c.get('op')!r}"
            if not feature_re().match(str(c.get("left", "")).strip()):
                return f"буруу feature {c.get('left')!r}"
            r = str(c.get("right", "")).strip()
            try:
                float(r)
            except ValueError:
                if not feature_re().match(r):
                    return f"буруу feature {r!r}"
    if not (strat.get("long_entry") or strat.get("short_entry")):
        return "оролтын дүрэм алга"
    return None


def signals(df: pd.DataFrame, strat: dict, cache: dict | None = None) -> pd.Series:
    cache = {} if cache is None else cache
    out = pd.Series(0, index=df.index, dtype=int)
    for side, val in (("long_entry", 1), ("short_entry", -1)):
        conds = strat.get(side) or []
        if conds:
            ok = pd.Series(True, index=df.index)
            for c in conds:
                ok &= _condition(df, c, cache)
            out[ok & (out == 0)] = val
    out.iloc[:200] = 0  # warm-up
    return out


# ---------- library file ----------

def rules_key(strat: dict) -> str:
    body = json.dumps({k: strat.get(k) for k in ("long_entry", "short_entry")}, sort_keys=True)
    return hashlib.sha1(body.encode()).hexdigest()[:12]


def load() -> list[dict]:
    if mkt().library_path.exists():
        return json.loads(mkt().library_path.read_text(encoding="utf-8")).get("strategies", [])
    return []


def save(strats: list[dict]) -> None:
    mkt().library_path.write_text(json.dumps({"strategies": strats}, indent=2, ensure_ascii=False), encoding="utf-8")


def approved() -> list[dict]:
    return [s for s in load() if s.get("status") == "approved"]


def add_candidates(cands: list[dict]) -> list[dict]:
    """Adds well-formed, not-yet-seen strategies as 'pending'. Returns the new ones."""
    lib = load()
    seen = {s["key"] for s in lib}
    new = []
    for c in cands:
        err = validate_rules(c)
        if err:
            log.info("Стратеги '%s' татгалзав: %s", c.get("name"), err)
            continue
        key = rules_key(c)
        if key in seen:
            continue
        seen.add(key)
        item = {"key": key, "name": str(c.get("name", "unnamed"))[:80],
                "description": str(c.get("description", ""))[:300], "source": str(c.get("source", ""))[:300],
                "long_entry": c.get("long_entry") or [], "short_entry": c.get("short_entry") or [],
                "status": "pending", "added": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        lib.append(item)
        new.append(item)
    save(lib)
    return new


def evaluate(frames: dict, strat: dict) -> dict:
    """Backtest the strategy alone, with the bot's standard stops and risk
    rules, on the 4 walk-forward windows and the vault."""
    def signal_fn(df):
        return pd.DataFrame({"signal": signals(df, strat), "atr": atr(df, 14)}, index=df.index)
    p = default_params()
    return {"walk_forward": validation.walk_forward(frames, p, signal_fn),
            "vault": validation.vault(frames, p, signal_fn)}


def review(frames: dict, only_pending: bool = True) -> list[dict]:
    """Backtests strategies and sets approved / rejected. To be approved a
    strategy must make money in at least 3 of 4 history windows AND in the
    vault (newest 15%)."""
    lib = load()
    changed = []
    for s in lib:
        if only_pending and s.get("status") != "pending":
            continue
        try:
            stats = evaluate(frames, s)
        except Exception as e:
            s["status"], s["error"] = "rejected", str(e)[:200]
            changed.append(s)
            continue
        wf, v = stats["walk_forward"], stats["vault"]
        ok = (wf["positive_folds"] >= 3 and wf["score"] > MIN_SCORE and v["score"] > 0
              and v["trades"] >= validation.FOLD_MIN_TRADES and v["return"] > 0)
        s["stats"], s["status"] = stats, "approved" if ok else "rejected"
        s["reviewed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        changed.append(s)
    save(lib)
    return changed


def ensure_seeded() -> None:
    if not mkt().library_path.exists():
        add_candidates(SEEDS)


def library_signal(df: pd.DataFrame, strats: list[dict] | None = None) -> pd.Series:
    """Average vote of approved strategies, in [-1, 1]."""
    strats = approved() if strats is None else strats
    if not strats:
        return pd.Series(0.0, index=df.index)
    cache = {}
    return sum(signals(df, s, cache) for s in strats) / len(strats)
