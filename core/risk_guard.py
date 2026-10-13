"""HARD risk limits.

The strategy and the self-improver never touch this file: they only write
strategy_params.json, and every order (live and backtest) goes through
check_entry() here. Change these numbers only by hand, deliberately.
"""
from dataclasses import dataclass

from .context import mkt

MAX_LEVERAGE = 2                  # notional across all positions <= equity * 2
RISK_PER_TRADE = 0.01             # lose at most 1% of equity if the stop is hit
MAX_DAILY_LOSS = 0.05             # -5% from the UTC day's start equity -> close all, stop for the day
LIVE_LIMITS = {                   # real money: strict
    "max_open_positions": 2,
    "max_total_open_risk": 0.025, # all open stops together may lose at most 2.5%
    "max_trades_per_day": 6,
}
PAPER_LIMITS = {                  # demo / signal: more trades = more to learn from (owner's choice, 2026-10-11)
    "max_open_positions": 10,
    "max_total_open_risk": 0.05,
    "max_trades_per_day": 20,
}
MIN_STOP_DISTANCE = 0.003         # stop closer than 0.3% = noise, refuse
MAX_STOP_DISTANCE = 0.08          # stop further than 8% = too wild, refuse

# Correlation: BTC and ETH move together, so two shorts are close to one double-size bet.
CORRELATED = 0.7                  # correlation above this counts as "the same bet"
CORRELATED_RISK_SCALE = 0.5       # a correlated same-direction trade gets half the risk

# Drawdown from the highest equity ever reached -> smaller bets.
DRAWDOWN_STEPS = [(0.10, 0.5), (0.20, 0.25)]   # (drawdown, risk multiplier)
DRAWDOWN_HALT = 0.30              # -30% from the peak: stop until you check it by hand (/resetpeak)

# Funding carry (spot long + perp short, market-neutral)
CARRY_MAX_FRACTION = 0.30         # at most 30% of equity in carry positions in total
CARRY_PER_POSITION = 0.10         # at most 10% of equity per carry position
CARRY_MAX_ADVERSE_MOVE = 0.40     # close a carry if price rose 40% (perp margin safety)

MIN_PAPER_DAYS_BEFORE_LIVE = 14

# A market whose prices move much less than crypto (FX: ~0.1%/hour) may set its own
# stop-distance and leverage bounds in its config.yaml "risk:" section, but never
# beyond these ceilings. Risk per trade, daily loss and drawdown rules stay global.
ABS_MAX_LEVERAGE = 10
ABS_MIN_STOP_DISTANCE = 0.0005


def market_bounds() -> tuple[float, float, float]:
    """(max_leverage, min_stop_distance, max_stop_distance) for the active market."""
    r = (getattr(mkt(), "cfg", None) or {}).get("risk") or {}
    lev = min(float(r.get("max_leverage", MAX_LEVERAGE)), ABS_MAX_LEVERAGE)
    lo = max(float(r.get("min_stop_distance", MIN_STOP_DISTANCE)), ABS_MIN_STOP_DISTANCE)
    hi = min(float(r.get("max_stop_distance", MAX_STOP_DISTANCE)), MAX_STOP_DISTANCE)
    return lev, lo, hi


def limits(mode: str | None = None) -> dict:
    """Position / total-risk / trades-per-day limits for a mode (default: the active market's)."""
    return LIVE_LIMITS if (mode or mkt().mode) == "live" else PAPER_LIMITS
LIVE_CONFIRM_PHRASE = "YES_I_ACCEPT_THE_RISK"


@dataclass
class Decision:
    allowed: bool
    qty: float = 0.0
    reason: str = ""
    risk_fraction: float = 0.0


def daily_loss_hit(day_start_equity: float, equity: float) -> bool:
    return day_start_equity > 0 and equity <= day_start_equity * (1 - MAX_DAILY_LOSS)


def drawdown_multiplier(equity: float, peak_equity: float) -> float:
    """1.0 normally, smaller after losses, 0.0 = halted."""
    if peak_equity <= 0:
        return 1.0
    dd = 1 - equity / peak_equity + 1e-9  # +1e-9: exactly -10.000% must count as -10% (float rounding)
    if dd >= DRAWDOWN_HALT:
        return 0.0
    mult = 1.0
    for level, m in DRAWDOWN_STEPS:
        if dd >= level:
            mult = m
    return mult


def check_entry(
    *,
    equity: float,
    entry: float,
    stop: float,
    side: int,
    open_positions: int,
    open_notional: float,
    trades_today: int,
    halted: bool,
    peak_equity: float = 0.0,
    open_risk: float = 0.0,
    max_corr_same_side: float = 0.0,
    blackout: str | None = None,
    paused: bool = False,
    risk_per_trade: float | None = None,
) -> Decision:
    """side: +1 long, -1 short. Returns the allowed quantity (base units)."""
    if paused:
        return Decision(False, reason="таны /stop командаар түр зогссон")
    if halted:
        return Decision(False, reason="kill switch идэвхтэй (өдрийн алдагдлын хязгаар)")
    if blackout:
        return Decision(False, reason=f"чухал үйл явдлын үе: {blackout}")
    if equity <= 0 or entry <= 0:
        return Decision(False, reason="equity/үнэ буруу")
    lim = limits()
    if open_positions >= lim["max_open_positions"]:
        return Decision(False, reason=f"нээлттэй position {lim['max_open_positions']}-д хүрсэн")
    if trades_today >= lim["max_trades_per_day"]:
        return Decision(False, reason=f"өдрийн арилжааны тоо {lim['max_trades_per_day']}-д хүрсэн")
    if side not in (1, -1):
        return Decision(False, reason="чиглэл буруу")
    # The stop must sit on the losing side of the entry.
    if (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
        return Decision(False, reason="stop-loss буруу талд байна")

    max_leverage, min_stop, max_stop = market_bounds()
    stop_dist = abs(entry - stop)
    stop_pct = stop_dist / entry
    if stop_pct < min_stop:
        return Decision(False, reason=f"stop хэт ойрхон ({stop_pct:.2%})")
    if stop_pct > max_stop:
        return Decision(False, reason=f"stop хэт хол ({stop_pct:.2%})")

    dd_mult = drawdown_multiplier(equity, peak_equity)
    if dd_mult == 0.0:
        return Decision(False, reason=f"оргилоосоо {DRAWDOWN_HALT:.0%} унасан тул зогслоо. "
                                      "Шалгаад /resetpeak гэж бичнэ үү")
    # a person may choose LESS risk than the global 1%, never more
    base = min(RISK_PER_TRADE, risk_per_trade) if risk_per_trade else RISK_PER_TRADE
    risk_frac = base * dd_mult
    if max_corr_same_side >= CORRELATED:
        risk_frac *= CORRELATED_RISK_SCALE
    risk_frac = min(risk_frac, lim["max_total_open_risk"] - open_risk / equity)
    if risk_frac <= RISK_PER_TRADE * 0.1:
        return Decision(False, reason="нийт нээлттэй эрсдэл хязгаартаа хүрсэн")

    qty = equity * risk_frac / stop_dist
    max_notional = equity * max_leverage - open_notional
    if max_notional <= 0:
        return Decision(False, reason="leverage хязгаар дүүрсэн")
    qty = min(qty, max_notional / entry)
    return Decision(True, qty=qty, reason="ok", risk_fraction=risk_frac)


def check_carry(*, equity: float, carry_notional: float, notional: float) -> Decision:
    if notional > equity * CARRY_PER_POSITION + 1e-9:
        return Decision(False, reason="нэг carry position хэт том")
    if carry_notional + notional > equity * CARRY_MAX_FRACTION:
        return Decision(False, reason=f"carry нийт {CARRY_MAX_FRACTION:.0%} хязгаарт хүрсэн")
    return Decision(True, reason="ok")


def live_trading_allowed(paper_days: float, confirm: str) -> tuple[bool, str]:
    if paper_days < MIN_PAPER_DAYS_BEFORE_LIVE:
        return False, (f"Бодит мөнгөөр арилжаалахаас өмнө demo/testnet дээр {MIN_PAPER_DAYS_BEFORE_LIVE} "
                       f"хоног ажиллах ёстой. Одоогоор: {paper_days:.1f} хоног.")
    if confirm != LIVE_CONFIRM_PHRASE:
        return False, f".env дотор LIVE_CONFIRM={LIVE_CONFIRM_PHRASE} гэж бичээгүй байна."
    return True, "ok"
