"""Shadow test: candidate settings that passed the walk-forward + vault
checks don't go live straight away. They "trade" alongside the live settings
on every new candle from the day they were created (a true forward test on
data no backtest has ever seen). One takes over only after SHADOW_MIN_DAYS if
it clearly beat the live settings over exactly the same period."""
import logging
import uuid

from . import validation
from .settings import SHADOW_MIN_DAYS
from .storage import now_ms
from .strategy import clamp_params, load_params, save_params

log = logging.getLogger(__name__)
DAY_MS = 86_400_000
MAX_SHADOWS = 6
MAX_AGE_DAYS = SHADOW_MIN_DAYS + 30   # a shadow gets a month past the minimum to collect enough trades
MIN_FORWARD_TRADES = 6
PROMOTE_MARGIN = 0.3


def get(st) -> list[dict]:
    return st.get("shadows", [])


def add(st, params: dict, metrics: dict) -> dict | None:
    """None when all places are taken: a newer candidate never pushes out one that is
    still running its forward test (with a month-long test it would never finish)."""
    shadows = get(st)
    if len(shadows) >= MAX_SHADOWS:
        return None
    s = {"id": uuid.uuid4().hex[:8], "created": now_ms(), "params": params, "backtest": metrics}
    shadows.append(s)
    st.set("shadows", shadows)
    return s


def review(st, frames: dict) -> dict:
    """Score every shadow against the live settings on the same forward
    window; promote one if it earned it; drop old or failing ones."""
    shadows = get(st)
    live = load_params()
    keep, report, winner = [], [], None
    for s in shadows:
        days = (now_ms() - s["created"]) / DAY_MS
        # settings saved before a market gained new sources lack their keys: fill the
        # defaults (crypto shadows from v1 crashed every daily review, 2026-10-20/21)
        s["params"] = clamp_params(s["params"])
        mine = validation.forward(frames, s["params"], s["created"])
        theirs = validation.forward(frames, live, s["created"])
        s["forward"], s["live_forward"], s["days"] = mine, theirs, round(days, 1)
        report.append({"id": s["id"], "days": round(days, 1), "shadow": mine, "live": theirs})
        ready = days >= SHADOW_MIN_DAYS and mine["trades"] >= MIN_FORWARD_TRADES
        if ready and mine["score"] > theirs["score"] + PROMOTE_MARGIN and mine["return"] > 0:
            if winner is None or mine["score"] > winner["forward"]["score"]:
                winner = s
            continue
        if days > MAX_AGE_DAYS or (days >= SHADOW_MIN_DAYS and mine["return"] < -0.05):
            log.info("Shadow %s хасагдлаа (%.0f хоног, forward %s)", s["id"], days, mine)
            continue
        keep.append(s)
    promoted = None
    if winner:
        save_params(winner["params"], {"backtest": winner["backtest"], "forward": winner["forward"]},
                    note=f"shadow {winner['id']} won forward test after {winner['days']} days")
        promoted = winner
        keep = [s for s in keep if s["id"] != winner["id"]]
    st.set("shadows", keep)
    return {"promoted": promoted, "shadows": report}
