"""Funding carry: buy the coin on spot AND short the same amount on the
perpetual. Price moves cancel out; what's left is the funding that longs pay
shorts (often 10-30%+ a year in bullish, crowded markets).

Risks that remain: funding can turn negative (we then exit), fees on 4 legs,
and the perp short needs margin if price rockets (1x leverage + an exit if
price rises 40%).
"""
import logging

import numpy as np

from core import risk_guard
from core.context import mkt
from core.notifier import notify
from core.storage import Storage, now_ms

from .data import public_exchange


def _cfg() -> dict:
    return mkt().carry_cfg

log = logging.getLogger(__name__)
PERIODS_PER_YEAR = 3 * 365      # Binance funding every 8h
LOOKBACK = 3                     # use the average of the last 3 funding payments
ROUND_TRIP_FEE = 2 * (0.001 + 0.0005)


def apr(rates) -> float:
    return float(np.mean(rates)) * PERIODS_PER_YEAR if len(rates) else 0.0


def recent_rates(symbol: str, n: int = LOOKBACK) -> list[float]:
    rows = public_exchange().fetch_funding_rate_history(symbol, limit=n)
    return [r["fundingRate"] for r in rows[-n:]]


def scan() -> list[dict]:
    """Current funding APR for each carry candidate, best first."""
    out = []
    for s in _cfg()["symbols"]:
        try:
            out.append({"symbol": s, "apr": round(apr(recent_rates(s)), 4)})
        except Exception as e:
            log.debug("carry scan %s: %s", s, e)
    return sorted(out, key=lambda x: -x["apr"])


def manage(st: Storage, broker, equity: float, halted: bool, allow_new: bool = True, user: str | None = None) -> None:
    """Called every hour: collect funding, exit weak carries, maybe open new ones."""
    open_c = {c["symbol"]: c for c in st.open_carries(mkt().mode, user)}

    for sym, c in open_c.items():
        # 1) book funding received
        amount, last = broker.carry_funding(sym, c["last_funding_ts"] or c["opened_ts"])
        if amount:
            st.update_carry(c["id"], funding=(c["funding"] or 0) + amount, last_funding_ts=last)
            c["funding"] = (c["funding"] or 0) + amount
        # 2) both legs still there? a broken hedge is an unhedged bet -> close
        legs = broker.carry_legs(sym)
        broken = legs["perp_qty"] < c["qty"] * 0.9 or legs["spot_qty"] < c["qty"] * 0.9
        px = broker.price(sym)
        rate_apr = apr(recent_rates(sym))
        reason = None
        if broken:
            reason = "hedge broken"
        elif px >= c["perp_entry"] * (1 + risk_guard.CARRY_MAX_ADVERSE_MOVE):
            reason = f"price rose {risk_guard.CARRY_MAX_ADVERSE_MOVE:.0%} (margin protection)"
        elif rate_apr < _cfg()["exit_apr"]:
            reason = f"funding fell to {rate_apr:.1%} per year"
        elif halted:
            reason = "kill switch"
        if reason:
            close(st, broker, c, reason, user)

    if halted or not allow_new:
        return
    carry_notional = sum(c["qty"] * c["perp_entry"] for c in st.open_carries(mkt().mode, user))
    for cand in scan():
        if cand["apr"] < _cfg()["min_apr"] or cand["symbol"] in open_c:
            continue
        notional = equity * risk_guard.CARRY_PER_POSITION
        dec = risk_guard.check_carry(equity=equity, carry_notional=carry_notional, notional=notional)
        if not dec.allowed:
            break
        try:
            r = broker.carry_open(cand["symbol"], notional)
        except Exception as e:
            notify(to=user, text=f"❌ Carry {cand['symbol']}: could not open: {e}", key=f"carryerr{cand['symbol']}", every_sec=21600)
            continue
        st.insert_carry({"mode": mkt().mode, "user": user, "symbol": cand["symbol"], "qty": r["qty"], "spot_entry": r["spot_px"],
                         "perp_entry": r["perp_px"], "opened_ts": now_ms(), "status": "open", "fees": r["fees"],
                         "funding": 0.0, "last_funding_ts": now_ms()})
        carry_notional += notional
        notify(to=user, text="\n".join(["🏦 CARRY OPENED", f"{cand['symbol'].split('/')[0]} · spot buy + futures short", "",
                          f"Size:     {notional:,.0f} $", f"Funding:  ≈ {cand['apr']:.1%} per year",
                          "Price moves cancel out; the funding income is what remains"]))


def close(st: Storage, broker, c: dict, reason: str, user: str | None = None) -> None:
    r = broker.carry_close(c["symbol"], c["qty"])
    if r is None:
        r = {"spot_px": c["spot_entry"], "perp_px": c["perp_entry"], "fees": 0.0}
    price_pnl = c["qty"] * (r["spot_px"] - c["spot_entry"]) + c["qty"] * (c["perp_entry"] - r["perp_px"])
    fees = (c["fees"] or 0) + r["fees"]
    pnl = price_pnl + (c["funding"] or 0) - fees
    st.update_carry(c["id"], status="closed", closed_ts=now_ms(), pnl=pnl, fees=fees, note=reason)
    notify(to=user, text="\n".join(["🏦 CARRY CLOSED", f"{c['symbol'].split('/')[0]}", "", f"Reason:   {reason}",
                      f"Funding:  {c['funding'] or 0:+.2f} $", f"Fees:     {-fees:+.2f} $", f"Total:    {pnl:+.2f} $"]))


def backtest(symbols=None, days: int = 730) -> list[dict]:
    """Replays the entry/exit rules on funding history (hedged, so price
    moves are ignored; fees on all 4 legs are charged)."""
    ex = public_exchange()
    out = []
    for s in symbols or _cfg()["symbols"]:
        since = now_ms() - days * 86_400_000
        rates = []
        while True:
            rows = ex.fetch_funding_rate_history(s, since=since, limit=1000)
            if not rows:
                break
            rates += [r["fundingRate"] for r in rows]
            since = rows[-1]["timestamp"] + 1
            if len(rows) < 1000:
                break
        if len(rates) < 30:
            continue
        inside, income, fees, periods, trades = False, 0.0, 0.0, 0, 0
        for i in range(LOOKBACK, len(rates)):
            a = apr(rates[i - LOOKBACK:i])   # decide with what was known before this payment
            if not inside and a >= _cfg()["min_apr"]:
                inside, trades = True, trades + 1
                fees += ROUND_TRIP_FEE
            elif inside and a < _cfg()["exit_apr"]:
                inside = False
            if inside:
                income += rates[i]
                periods += 1
        years = len(rates) / PERIODS_PER_YEAR
        out.append({"symbol": s, "years": round(years, 2), "entries": trades,
                    "time_in_market": round(periods / max(1, len(rates) - LOOKBACK), 3),
                    "return_per_year_on_notional": round((income - fees) / years, 4),
                    # spot purchase + 1x perp margin = 2x the notional tied up
                    "return_per_year_on_capital": round((income - fees) / years / 2, 4)})
    return out
