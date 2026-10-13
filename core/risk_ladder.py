"""Risk ladder (owner's idea, 2026-10-13): risk per trade starts low and only
steps up after the bot has proven itself, judged every `window` closed trades:

  last 30 trades: average R > 0 and profit factor > 1.2  -> one step up
  last 30 trades: average R < 0                          -> one step down
  otherwise                                               -> stay, look at the next 30

Judged on R (profit per unit of risk) and profit factor, not on win rate: a bot
can win 70% of its trades and still lose money. The top step never exceeds
risk_guard.RISK_PER_TRADE, and a market without a "risk.ladder" config has no ladder.
"""
import numpy as np

from .context import mkt
from .risk_guard import RISK_PER_TRADE


def config() -> dict | None:
    lad = ((getattr(mkt(), "cfg", None) or {}).get("risk") or {}).get("ladder")
    if not lad:
        return None
    return {"levels": [min(float(x), RISK_PER_TRADE) for x in lad.get("levels", [RISK_PER_TRADE])],
            "window": int(lad.get("window", 30)), "up_avg_r": float(lad.get("up_avg_r", 0.0)),
            "up_profit_factor": float(lad.get("up_profit_factor", 1.2)),
            "down_avg_r": float(lad.get("down_avg_r", 0.0))}


def profit_factor(rs) -> float:
    rs = np.asarray(rs, dtype=float)
    loss = -rs[rs < 0].sum()
    return float(rs[rs > 0].sum() / loss) if loss > 0 else float("inf")


def step(state: dict, closed_rs_newest_first: list, cfg: dict) -> tuple[dict, str | None]:
    """New ladder state and a message if the level changed."""
    levels = cfg["levels"]
    lvl = min(int(state.get("level", 0)), len(levels) - 1)
    judged = int(state.get("judged", 0))
    total = len(closed_rs_newest_first)
    if total - judged < cfg["window"]:
        return {"level": lvl, "judged": judged}, None
    rs = closed_rs_newest_first[:cfg["window"]]
    avg, pf = float(np.mean(rs)), profit_factor(rs)
    new = lvl
    if avg > cfg["up_avg_r"] and pf > cfg["up_profit_factor"]:
        new = min(lvl + 1, len(levels) - 1)
    elif avg < cfg["down_avg_r"]:
        new = max(lvl - 1, 0)
    msg = None
    if new != lvl:
        arrow = "⬆️" if new > lvl else "⬇️"
        msg = (f"{arrow} Risk per trade {levels[lvl]:.2%} -> {levels[new]:.2%} "
               f"(last {cfg['window']} trades: avg {avg:+.2f}R, profit factor {pf:.2f})")
    return {"level": new, "judged": total}, msg


def current(state: dict, cfg: dict) -> float:
    return cfg["levels"][min(int(state.get("level", 0)), len(cfg["levels"]) - 1)]
