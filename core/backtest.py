"""Bar-by-bar simulator. Uses the same risk_guard and exits.py as live
trading, and charges fees, slippage and funding so results are not flattering.

Entry (market entry_order = maker): a limit order `entry_offset_atr` ATRs better than
the signal close, valid for one candle; filled only if price trades there.
Stops are market orders (taker fee), take-profits are limit orders (maker fee).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import exits, risk_guard
from .reward import max_drawdown, strategy_score
from .context import mkt
from .strategy import compute

DAY_MS = 86_400_000


@dataclass
class Result:
    trades: list = field(default_factory=list)
    equity: np.ndarray = field(default_factory=lambda: np.array([]))
    start_equity: float = 1000.0
    missed: int = 0  # limit entries that never filled

    @property
    def r(self):
        return [t["r"] for t in self.trades]

    @property
    def total_return(self):
        return float(self.equity[-1] / self.start_equity - 1) if len(self.equity) else 0.0

    @property
    def max_dd(self):
        return max_drawdown(self.equity)

    def summary(self) -> dict:
        n = len(self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        reasons = {}
        for t in self.trades:
            reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
        return {
            "trades": n,
            "win_rate": round(wins / n, 3) if n else 0.0,
            "avg_r": round(float(np.mean(self.r)), 3) if n else 0.0,
            "return": round(self.total_return, 4),
            "max_dd": round(self.max_dd, 4),
            "longs": sum(1 for t in self.trades if t["side"] == 1),
            "shorts": sum(1 for t in self.trades if t["side"] == -1),
            "missed_entries": self.missed,
            "exits": reasons,
            "score": round(strategy_score(self.r, self.max_dd), 3),
        }


def run(df: pd.DataFrame, params: dict, start: int = 0, end: int | None = None,
        feats: pd.DataFrame | None = None, start_equity: float = 1000.0,
        entry_order: str | None = None, costs: dict | None = None) -> Result:
    """Simulate on rows [start, end). Features are computed on the full frame
    first so indicators are warmed up; nothing looks into the future."""
    if feats is None:
        feats = compute(df, params)
    end = len(df) if end is None else end
    ts = df.index.to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    fund = df["hold_cost"].to_numpy()  # e.g. crypto funding: longs pay positive rates
    costs = costs or mkt().costs
    MAKER_FEE, TAKER_FEE, SLIPPAGE = costs["maker"], costs["taker"], costs["slippage"]
    entry_order = entry_order or mkt().entry_order
    sig = feats["signal"].to_numpy()
    atr = feats["atr"].to_numpy()
    maker = entry_order == "maker"
    offset = params.get("entry_offset_atr", 0.0) if maker else 0.0

    equity = peak = start_equity
    pos = None
    day, day_start, halted, trades_today = None, equity, False, 0
    trades, curve, missed = [], [], 0
    max_hold = int(params["max_hold_bars"])

    def close_part(qty, px, fee_rate, reason=None):
        nonlocal equity
        gross = pos["side"] * qty * (px - pos["entry"])
        fee = qty * px * fee_rate
        equity += gross - fee
        pos["realized"] += gross - fee
        pos["qty"] -= qty
        if reason:
            net = pos["realized"] - pos["cost"]
            trades.append({"ts": int(ts[i]), "side": pos["side"], "entry": pos["entry"], "exit": px,
                           "pnl": net, "r": net / pos["risk"], "reason": reason})

    for i in range(max(start, 1), end):
        d = ts[i] // DAY_MS
        if d != day:
            day, day_start, halted, trades_today = d, equity, False, 0

        # ---- entry on the previous candle's signal ----
        if pos is None and sig[i - 1] != 0 and not halted:
            side = int(sig[i - 1])
            dist = params["stop_atr"] * atr[i - 1]
            if maker:
                limit = c[i - 1] - side * offset * atr[i - 1]
                # a post-only order can't sit on the wrong side of the market, so if the
                # candle opens through the limit we join the book at the open instead
                limit = min(limit, o[i]) if side == 1 else max(limit, o[i])
                filled = l[i] <= limit if side == 1 else h[i] >= limit
                entry, fee_rate = limit, MAKER_FEE
            else:
                filled, entry, fee_rate = True, o[i] * (1 + SLIPPAGE * side), TAKER_FEE
            if not filled:
                missed += 1
            else:
                stop = entry - side * dist
                dec = risk_guard.check_entry(equity=equity, entry=entry, stop=stop, side=side, open_positions=0,
                                             open_notional=0.0, trades_today=trades_today, halted=halted,
                                             peak_equity=peak)
                if dec.allowed:
                    entry_fee = dec.qty * entry * fee_rate
                    equity -= entry_fee
                    trades_today += 1
                    pos = {"side": side, "qty": dec.qty, "entry": entry, "stop": stop, "dist": dist,
                           "tp": entry + side * dist * params["tp_r"],
                           "partial": exits.partial_price(entry, side, dist, params), "best": entry,
                           "bar": i, "risk": dec.qty * dist, "cost": entry_fee, "realized": 0.0}

        # ---- manage the open position on this candle ----
        if pos is not None:
            side = pos["side"]
            if fund[i] and i > pos["bar"]:  # longs pay positive funding, shorts receive it
                f = side * pos["qty"] * o[i] * fund[i]
                equity -= f
                pos["cost"] += f
            stop_hit = l[i] <= pos["stop"] if side == 1 else h[i] >= pos["stop"]
            if stop_hit:  # pessimistic: if stop and target are both inside the candle, assume the stop came first
                px = (min(o[i], pos["stop"]) if side == 1 else max(o[i], pos["stop"])) * (1 - SLIPPAGE * side)
                close_part(pos["qty"], px, TAKER_FEE, "stop" if pos["realized"] == 0 else "stop_after_partial")
                pos = None
            else:
                if pos["partial"] is not None and (h[i] >= pos["partial"] if side == 1 else l[i] <= pos["partial"]):
                    close_part(pos["qty"] / 2, pos["partial"], MAKER_FEE)
                    pos["partial"] = None
                if h[i] >= pos["tp"] if side == 1 else l[i] <= pos["tp"]:
                    close_part(pos["qty"], pos["tp"], MAKER_FEE, "tp")
                    pos = None
                elif i - pos["bar"] >= max_hold:
                    close_part(pos["qty"], c[i] * (1 - SLIPPAGE * side), TAKER_FEE, "time")
                    pos = None
                elif sig[i] == -side:
                    close_part(pos["qty"], c[i] * (1 - SLIPPAGE * side), TAKER_FEE, "flip")
                    pos = None
                else:
                    pos["stop"] = exits.update_stop(pos, h[i], l[i], c[i], atr[i], params)
            if pos is None and risk_guard.daily_loss_hit(day_start, equity):
                halted = True

        unreal = pos["side"] * pos["qty"] * (c[i] - pos["entry"]) if pos else 0.0
        curve.append(equity + unreal)
        peak = max(peak, equity)

    return Result(trades=trades, equity=np.array(curve), start_equity=start_equity, missed=missed)
