"""Exit management shared by the backtester and live trading, so both behave
the same: move the stop to breakeven, trail it, take half off early.

A setting below its "off" threshold disables that feature, so the
self-improver can switch features on or off just by tuning numbers."""

BE_OFF, TRAIL_OFF, PARTIAL_OFF = 0.3, 0.5, 0.5
BE_LOCK_R = 0.1   # breakeven stop sits slightly in profit to cover fees


def partial_price(entry: float, side: int, dist: float, p: dict) -> float | None:
    if p["partial_r"] < PARTIAL_OFF:
        return None
    return entry + side * dist * p["partial_r"]


def update_stop(pos: dict, high: float, low: float, close: float, atr: float, p: dict) -> float:
    """Call once per closed candle. pos needs side, entry, stop, dist, best.
    Updates pos["best"] and returns the new stop (never looser than before)."""
    side = pos["side"]
    pos["best"] = max(pos["best"], high) if side == 1 else min(pos["best"], low)
    tighter = max if side == 1 else min
    new = pos["stop"]
    gain_r = (pos["best"] - pos["entry"]) * side / pos["dist"]
    if p["be_at_r"] >= BE_OFF and gain_r >= p["be_at_r"]:
        new = tighter(new, pos["entry"] + side * BE_LOCK_R * pos["dist"])
    if p["trail_atr"] >= TRAIL_OFF:
        new = tighter(new, close - side * p["trail_atr"] * atr)
    return new
