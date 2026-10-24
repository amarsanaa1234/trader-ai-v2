"""Portfolio backtest: every symbol trades from ONE shared account, the way the
live bot does. backtest.py judges each symbol alone with its own money, which
is fast for ranking thousands of settings but ignores that positions compete
for the same equity, position slots, total-risk budget, leverage and (FX)
the per-currency cap. This simulator applies all of those together.

Same fills, fees, funding, exits and risk_guard as backtest.py.
"""
import numpy as np

from . import exits, risk_guard
from .backtest import entry_wait_candles
from .context import mkt
from .reward import max_drawdown, strategy_score
from .strategy import compute

DAY_MS = 86_400_000


def run(frames: dict, p: dict, start_ts: int | None = None, end_ts: int | None = None,
        start_equity: float = 10_000.0, feats: dict | None = None, risk_per_trade: float | None = None,
        with_curve: bool = False) -> dict:
    m = mkt()
    S = {}
    for sym, df in frames.items():
        f = (feats or {}).get(sym)
        if f is None:
            f = compute(df, p)
        cst = m.costs_for(sym)
        S[sym] = {"ts": df.index.to_numpy(), "o": df["open"].to_numpy(), "h": df["high"].to_numpy(),
                  "l": df["low"].to_numpy(), "c": df["close"].to_numpy(), "fund": df["hold_cost"].to_numpy(),
                  "sig": f["signal"].to_numpy(), "atr": f["atr"].to_numpy(),
                  "force": np.asarray(m.force_exit(df.index, p), dtype=bool),
                  "maker": cst["maker"], "taker": cst["taker"], "slip": cst["slippage"],
                  "through": cst.get("fill_through", 0.0)}
        S[sym]["row"] = {int(t): i for i, t in enumerate(S[sym]["ts"])}
    timeline = sorted({int(t) for s in S.values() for t in s["ts"]
                       if (start_ts is None or t >= start_ts) and (end_ts is None or t < end_ts)})
    maker_entry = m.entry_order == "maker"
    offset = p.get("entry_offset_atr", 0.0) if maker_entry else 0.0
    max_hold = int(p["max_hold_bars"])
    wait = entry_wait_candles()

    equity = peak = start_equity
    pos, pend, last_close = {}, {}, {}
    day, day_start, halted, trades_today = None, equity, False, 0
    trades, curve, refused = [], [], {}

    def close_part(sym, qty, px, fee_rate, reason=None, t=0):
        nonlocal equity
        q = pos[sym]
        gross = q["side"] * qty * (px - q["entry"])
        fee = qty * px * fee_rate
        equity += gross - fee
        q["realized"] += gross - fee
        q["qty"] -= qty
        if reason:
            net = q["realized"] - q["cost"]
            trades.append({"symbol": sym, "ts": t, "side": q["side"], "pnl": net, "r": net / q["risk"],
                           "reason": reason})
            del pos[sym]

    for t in timeline:
        d = t // DAY_MS
        if d != day:
            day, day_start, halted, trades_today = d, equity, False, 0
        for sym, s in S.items():
            i = s["row"].get(t)
            if i is None or i < 1:
                continue
            o, h, l, c = s["o"][i], s["h"][i], s["l"][i], s["c"][i]
            # ---- entry on the previous candle's signal (a limit may wait `wait` candles) ----
            if sym not in pos and sym not in pend and s["sig"][i - 1] != 0 and not halted:
                sd = int(s["sig"][i - 1])
                pend[sym] = {"side": sd, "dist": p["stop_atr"] * s["atr"][i - 1],
                             "limit": s["c"][i - 1] - sd * offset * s["atr"][i - 1], "until": i + wait - 1}
            if sym not in pos and sym in pend:
                side, dist = pend[sym]["side"], pend[sym]["dist"]
                if maker_entry:
                    limit = min(pend[sym]["limit"], o) if side == 1 else max(pend[sym]["limit"], o)
                    th = s["through"]
                    filled = l <= limit * (1 - th) if side == 1 else h >= limit * (1 + th)
                    entry, fee_rate = limit, s["maker"]
                else:
                    filled, entry, fee_rate = True, o * (1 + s["slip"] * side), s["taker"]
                if filled or i >= pend[sym]["until"] or halted:
                    pend.pop(sym)
                if filled and np.isfinite(dist) and dist > 0:
                    legs = [(k, q["side"], q["qty"] * q["dist"] / equity) for k, q in pos.items()]
                    dec = risk_guard.check_entry(
                        equity=equity, entry=entry, stop=entry - side * dist, side=side, open_positions=len(pos),
                        open_notional=sum(q["qty"] * q["entry"] for q in pos.values()), trades_today=trades_today,
                        halted=halted, peak_equity=peak, risk_per_trade=risk_per_trade,
                        open_risk=sum(q["qty"] * max(0.0, q["side"] * (q["entry"] - q["stop"])) for q in pos.values()),
                        same_currency_risk=risk_guard.currency_risk(legs, sym, side)
                        if risk_guard.max_currency_risk() is not None else 0.0)
                    if dec.allowed:
                        fee = dec.qty * entry * fee_rate
                        equity -= fee
                        trades_today += 1
                        pos[sym] = {"side": side, "qty": dec.qty, "entry": entry, "stop": entry - side * dist,
                                    "dist": dist, "tp": entry + side * dist * p["tp_r"],
                                    "partial": exits.partial_price(entry, side, dist, p), "best": entry,
                                    "bar": i, "risk": dec.qty * dist, "cost": fee, "realized": 0.0}
                    else:
                        refused[dec.reason] = refused.get(dec.reason, 0) + 1
            # ---- manage the open position on this candle ----
            q = pos.get(sym)
            if q is not None:
                side, slip, taker = q["side"], s["slip"], s["taker"]
                if s["fund"][i] and i > q["bar"]:
                    f = side * q["qty"] * o * s["fund"][i]
                    equity -= f
                    q["cost"] += f
                if (l <= q["stop"]) if side == 1 else (h >= q["stop"]):
                    px = (min(o, q["stop"]) if side == 1 else max(o, q["stop"])) * (1 - slip * side)
                    close_part(sym, q["qty"], px, taker, "stop" if q["realized"] == 0 else "stop_after_partial", t)
                else:
                    if q["partial"] is not None and ((h >= q["partial"]) if side == 1 else (l <= q["partial"])):
                        close_part(sym, q["qty"] / 2, q["partial"], s["maker"])
                        q["partial"] = None
                    if (h >= q["tp"]) if side == 1 else (l <= q["tp"]):
                        close_part(sym, q["qty"], q["tp"], s["maker"], "tp", t)
                    elif i - q["bar"] >= max_hold:
                        close_part(sym, q["qty"], c * (1 - slip * side), taker, "time", t)
                    elif s["sig"][i] == -side:
                        close_part(sym, q["qty"], c * (1 - slip * side), taker, "flip", t)
                    elif s["force"][i]:
                        close_part(sym, q["qty"], c * (1 - slip * side), taker, "weekend", t)
                    else:
                        q["stop"] = exits.update_stop(q, h, l, c, s["atr"][i], p)
            last_close[sym] = c
        unreal = sum(q["side"] * q["qty"] * (last_close[k] - q["entry"]) for k, q in pos.items())
        if not halted and risk_guard.daily_loss_hit(day_start, equity + unreal):
            # live kill switch: close everything, no new trades until tomorrow
            for k in list(pos):
                close_part(k, pos[k]["qty"], last_close[k] * (1 - S[k]["slip"] * pos[k]["side"]),
                           S[k]["taker"], "kill_switch", t)
            halted, unreal = True, 0.0
        curve.append(equity + unreal)
        peak = max(peak, equity)

    curve = np.asarray(curve)
    rs = [x["r"] for x in trades]
    dd = max_drawdown(curve) if len(curve) else 0.0
    extra = {"curve": curve, "times": np.asarray(timeline), "trade_times": [x["ts"] for x in trades]} \
        if with_curve else {}
    return {**extra, "return": round(float(curve[-1] / start_equity - 1), 4) if len(curve) else 0.0,
            "max_dd": round(dd, 4), "trades": len(trades),
            "avg_r": round(float(np.mean(rs)), 3) if rs else 0.0,
            "score": round(strategy_score(rs, dd), 3),
            "per_symbol": {k: sum(1 for x in trades if x["symbol"] == k) for k in frames},
            "refused": dict(sorted(refused.items(), key=lambda x: -x[1])[:5])}


def vault(frames: dict, p: dict) -> dict:
    """The shared-account result on the vault period (newest 15% of history)."""
    from .validation import research_end
    cut = max(int(df.index[research_end(len(df))]) for df in frames.values())
    return run(frames, p, start_ts=cut)
