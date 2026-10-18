"""Stop rule for demo/live: does the bot trade as well as its backtest promised?

After at least MIN_TRADES closed trades with the current settings, the live
average R is compared with the vault's. If it is clearly worse (more than two
standard errors below) AND losing, the person's account is paused and told
why: either the market changed or the backtest was too optimistic. A human
looks before it trades again (/resume).
"""
import json

import numpy as np

from .context import mkt

MIN_TRADES = 30
Z = 2.0


def expected_avg_r() -> float | None:
    path = mkt().params_path
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    v = (data.get("metrics") or {}).get("vault") or {}
    return float(v["avg_r"]) if "avg_r" in v else None


def verdict(live_rs: list, expected: float | None) -> str | None:
    """A message if the live results are clearly worse than expected, else None."""
    if expected is None or len(live_rs) < MIN_TRADES:
        return None
    a = np.asarray(live_rs, dtype=float)
    se = a.std(ddof=1) / np.sqrt(len(a))
    if a.mean() < 0 and a.mean() < expected - Z * se:
        return (f"🛑 Paused: the last {len(a)} trades averaged {a.mean():+.2f}R, the backtest expected "
                f"{expected:+.2f}R (clearly worse, not just bad luck). The market may have changed or the "
                f"backtest was too optimistic. Check it, then /resume.")
    return None


def since_settings_changed() -> int:
    """Only trades opened with the current settings count."""
    path = mkt().params_path
    return int(path.stat().st_mtime * 1000) if path.exists() else 0
