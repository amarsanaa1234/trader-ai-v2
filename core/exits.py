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


# Take-profit extension (owner, 2026-11-15) - a FIXED rule for the markets whose config says
# tp_extend: true; the self-improver and the shadows can't tune or switch it.
# When the price comes within 0.1R of the TP, the bot asks its score again (last closed
# candle). If it still points the trade's way at least half as strongly as needed to enter,
# the stop moves to 0.15R behind the TP and the TP moves 1R further - checked again every time.
TP_NEAR_R, TP_LOCK_R, TP_EXT_R, TP_MIN_SCORE = 0.1, 0.15, 1.0, 0.5


def tp_extension(side: int, tp: float, stop: float, dist: float, price: float, score: float | None,
                 threshold: float) -> tuple[float, float] | None:
    """(new_tp, new_stop) if the TP should be pushed out now, else None. price = the best
    price reached (high for a long, low for a short)."""
    if score is None or score != score or not dist or side * (price - tp) < -TP_NEAR_R * dist:
        return None
    if side * score < TP_MIN_SCORE * threshold:
        return None
    lock = tp - side * TP_LOCK_R * dist
    new_stop = max(stop, lock) if side == 1 else min(stop, lock)
    return tp + side * TP_EXT_R * dist, new_stop


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
