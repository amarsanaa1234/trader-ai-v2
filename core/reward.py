"""The bot's "dopamine".

Rewards are about good decisions, not lucky profit:
  * reward is measured in R (profit divided by the risk taken), so betting
    bigger never earns more reward,
  * a disciplined loss (stop honoured, risk respected) is only mildly negative,
  * a missing stop-loss is punished hard.

Dopamine is a moving average of recent rewards. It never changes position
size (risk_guard owns that). It only steers how boldly the self-improver
explores new strategy settings: high dopamine = keep what works, low dopamine
= search wider.
"""
import numpy as np

DISCIPLINE_BONUS = 0.25
MISSING_STOP_PENALTY = -2.0
DOPAMINE_DECAY = 0.85


def trade_reward(r_multiple: float, exit_reason: str) -> float:
    r = float(np.clip(r_multiple, -2.0, 3.0))
    if exit_reason == "no_stop":
        return MISSING_STOP_PENALTY
    # plan followed: exited by its own stop / target (incl. a stop moved to breakeven or trailed)
    bonus = DISCIPLINE_BONUS if exit_reason in ("stop", "tp", "stop_after_partial", "trail") else 0.0
    return round(r + bonus, 4)


def update_dopamine(dopamine: float, reward: float) -> float:
    return round(DOPAMINE_DECAY * dopamine + (1 - DOPAMINE_DECAY) * reward, 4)


def mood(dopamine: float) -> str:
    if dopamine > 0.4:
        return "😄 урам орсон"
    if dopamine >= 0.0:
        return "🙂 тайван"
    if dopamine > -0.4:
        return "😐 болгоомжтой"
    return "😟 шинэ арга хайж байна"


def mood_en(dopamine: float) -> str:
    """English mood for Telegram (the dashboard keeps the Mongolian one)."""
    if dopamine > 0.4:
        return "😄 motivated"
    if dopamine >= 0.0:
        return "🙂 calm"
    if dopamine > -0.4:
        return "😐 cautious"
    return "😟 looking for new ideas"


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def strategy_score(r_multiples, max_dd: float, min_trades: int = 20) -> float:
    """Risk-adjusted score used to compare strategy settings (higher = better).
    Roughly a t-statistic of the per-trade R, minus a drawdown penalty, so a
    few lucky trades cannot win."""
    rs = np.asarray(r_multiples, dtype=float)
    n = len(rs)
    if n < min_trades:
        return -5.0 + n * 0.1
    t_stat = rs.mean() / (rs.std(ddof=1) + 1e-9) * np.sqrt(n)
    return float(t_stat - 8.0 * max_dd)
