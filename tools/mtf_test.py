"""Big timeframe for direction, small timeframe for the entry (owner's idea, 2026-10-26).

The 4h settings decide the DIRECTION exactly as now. Instead of a limit order, the
bot then watches 15-minute candles for up to 8 hours (the same wait as now) and only
enters when the small chart agrees with the big one:

  rejection  - 4h says short; a 15m candle pushes up to the 15m EMA20 (a pull-back)
               and then closes back down below it (a bearish close) -> short now.
               Mirror for longs.
  break      - 4h says short; after a small pull-back up (a higher 15m high in the
               last 3 candles) a 15m candle closes below the previous candle's low.

Stops:  atr   - same distance as now (stop_atr x the 4h ATR), only the price is better
        swing - just beyond the pull-back's extreme (highest 15m high of the last 4
                candles + 0.1 x 4h ATR); smaller risk per unit -> bigger size for the
                same 0.5% risk, but more easily stopped out

Everything else is the live logic: take-profit tp_r x stop distance, half taken at
partial_r, trailing / breakeven updated at each 4h close, flip on an opposite 4h
signal, max_hold_bars 4h candles, swap at each 4h candle, the market's per-symbol
costs (entries are market orders at the 15m close: taker fee + slippage), and the
same risk_guard sizing. Compared with the current limit-order backtest over the
same years, split into the research years and the vault (newest 15%).

    python tools/mtf_test.py <market> <m15.db> [strategy_params.json]
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[1]))

from core import backtest, exits, improver, risk_guard, validation  # noqa: E402
from core.reward import max_drawdown  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params, compute, load_params  # noqa: E402

M15 = 15 * 60_000
WAIT_MS = 8 * 3_600_000
START_EQ = 1000.0


def load15(path, sym):
    db = sqlite3.connect(path)
    d = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf='15m' ORDER BY ts",
                          db, params=(sym,))
    d["ema"] = d["close"].ewm(span=20, adjust=False).mean()
    return d


def trigger(d, j, side, kind):
    """Does 15m candle j (decided at its close) confirm an entry in `side`?"""
    o, h, l, c, e = d["o"][j], d["h"][j], d["l"][j], d["c"][j], d["e"][j]
    if kind == "rejection":
        if side == -1:
            return h >= d["e"][j - 1] and c < e and c < o
        return l <= d["e"][j - 1] and c > e and c > o
    # break: a pull-back against the trade in the last 3 candles, then a close beyond the previous candle
    if side == -1:
        pulled = any(d["h"][k] > d["h"][k - 1] for k in range(j - 3, j))
        return pulled and c < d["l"][j - 1]
    pulled = any(d["l"][k] < d["l"][k - 1] for k in range(j - 3, j))
    return pulled and c > d["h"][j - 1]


def simulate(df4, f4, d15, p, costs, kind, stop_kind, start_ts):
    """Small-timeframe entries on the 4h direction. Returns trades [(entry_ts, R, pnl)] and equity curve."""
    ts4 = df4.index.to_numpy()
    sig, atr = f4["signal"].to_numpy(), f4["atr"].to_numpy()
    h4, l4, c4, fund = (df4[k].to_numpy() for k in ("high", "low", "close", "hold_cost"))
    tf = ts4[1] - ts4[0]
    d = {"t": d15["ts"].to_numpy(), "o": d15["open"].to_numpy(), "h": d15["high"].to_numpy(),
         "l": d15["low"].to_numpy(), "c": d15["close"].to_numpy(), "e": d15["ema"].to_numpy()}
    t15 = d["t"]
    TAKER, MAKER, SLIP = costs["taker"], costs["maker"], costs["slippage"]
    _, min_stop, _ = risk_guard.market_bounds()
    equity = peak = START_EQ
    trades, curve = [], []
    i = int(np.searchsorted(ts4, start_ts))
    max_hold = int(p["max_hold_bars"])
    while i < len(ts4) - 1:
        s = int(sig[i - 1]) if i > 0 else 0
        if s == 0:
            curve.append(equity)
            i += 1
            continue
        t0 = ts4[i]                         # the signal candle closed at ts4[i] (= open of candle i)
        j = int(np.searchsorted(t15, t0))
        entry = None
        while j < len(t15) and t15[j] + M15 <= t0 + WAIT_MS:
            if j >= 4 and trigger(d, j, s, kind):
                entry, jt = d["c"][j] * (1 + SLIP * s), j
                break
            j += 1
        if entry is None:
            curve.append(equity)
            i += 1
            continue
        a = atr[i - 1]
        if stop_kind == "atr":
            dist = p["stop_atr"] * a
        else:
            ext = max(d["h"][jt - 3:jt + 1]) if s == -1 else min(d["l"][jt - 3:jt + 1])
            dist = abs(ext - entry) + 0.1 * a
        dist = max(dist, entry * min_stop * 1.01)
        dec = risk_guard.check_entry(equity=equity, entry=entry, stop=entry - s * dist, side=s, open_positions=0,
                                     open_notional=0.0, trades_today=0, halted=False, peak_equity=peak)
        if not dec.allowed:
            i += 1
            continue
        qty = dec.qty
        realized = -qty * entry * TAKER
        pos = {"side": s, "entry": entry, "stop": entry - s * dist, "dist": dist, "best": entry,
               "partial_done": False}
        tp = entry + s * dist * p["tp_r"]
        part = exits.partial_price(entry, s, dist, p)
        left = qty
        open_bar = int(np.searchsorted(ts4, t15[jt], side="right")) - 1   # 4h candle the entry is in
        k, exit_reason = jt + 1, None
        while k < len(t15) and exit_reason is None:
            o, h, l, c = d["o"][k], d["h"][k], d["l"][k], d["c"][k]
            if (l <= pos["stop"]) if s == 1 else (h >= pos["stop"]):
                px = (min(o, pos["stop"]) if s == 1 else max(o, pos["stop"])) * (1 - SLIP * s)
                realized += s * left * (px - entry) - left * px * TAKER
                exit_reason = "stop"
                break
            if part is not None and ((h >= part) if s == 1 else (l <= part)):
                half = left / 2
                realized += s * half * (part - entry) - half * part * MAKER
                left -= half
                part, pos["partial_done"] = None, True
            if (h >= tp) if s == 1 else (l <= tp):
                realized += s * left * (tp - entry) - left * tp * MAKER
                exit_reason = "tp"
                break
            # at each 4h close: swap, flip / time exit, trailing & breakeven (as live)
            if (t15[k] + M15) % tf == ts4[0] % tf:
                b = int(np.searchsorted(ts4, t15[k] + M15 - tf))
                if b < len(ts4) and ts4[b] == t15[k] + M15 - tf:
                    if fund[b]:
                        realized -= s * left * c4[b] * fund[b]
                    held = b - open_bar
                    if sig[b] == -s or held >= max_hold:
                        px = c * (1 - SLIP * s)
                        realized += s * left * (px - entry) - left * px * TAKER
                        exit_reason = "flip" if sig[b] == -s else "time"
                        break
                    pos["stop"] = exits.update_stop(pos, h4[b], l4[b], c4[b], atr[b], p)
            k += 1
        if exit_reason is None:      # data ended: close at the last price
            px = d["c"][min(k, len(t15) - 1)]
            realized += s * left * (px - entry) - left * px * TAKER
        equity += realized
        peak = max(peak, equity)
        trades.append((int(t15[jt]), realized / (qty * dist), realized))
        curve.append(equity)
        i = max(i + 1, int(np.searchsorted(ts4, t15[min(k, len(t15) - 1)], side="right")))
    return trades, np.array(curve)


def summarize(trades, curve, cut):
    out = {}
    for name, sel in (("research", lambda t: t[0] < cut), ("vault", lambda t: t[0] >= cut)):
        tr = [t for t in trades if sel(t)]
        rs = np.array([t[1] for t in tr]) if tr else np.array([])
        out[name] = {"n": len(tr), "win": float(np.mean(rs > 0)) if len(rs) else 0.0,
                     "avg_r": float(rs.mean()) if len(rs) else 0.0, "pnl": float(sum(t[2] for t in tr))}
    out["max_dd"] = max_drawdown(curve) if len(curve) else 0.0
    return out


def main():
    m15 = sys.argv[2]
    p = clamp_params(json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))["params"]) if len(sys.argv) > 3 \
        else load_params()
    frames = improver.load_frames(Storage(), mkt().symbols)
    variants = [("rejection", "atr"), ("rejection", "swing"), ("break", "atr"), ("break", "swing")]
    totals = {v: {"research": [], "vault": [], "dd": []} for v in ["now (4h limit)"] + [f"{a}/{b}" for a, b in variants]}
    for sym, df4 in frames.items():
        d15 = load15(m15, sym)
        if len(d15) < 1000:
            continue
        f4 = compute(df4, p)
        start_ts = int(d15["ts"].iloc[0]) + 20 * 24 * 3_600_000          # 15m warm-up
        start = int(np.searchsorted(df4.index.to_numpy(), start_ts))
        cut_i = validation.research_end(len(df4))
        cut = int(df4.index[cut_i])
        costs = mkt().costs_for(sym)
        for name, a, b in (("research", start, cut_i), ("vault", cut_i, len(df4))):
            res = backtest.run(df4, p, start=a, end=b, feats=f4, costs=costs)
            totals["now (4h limit)"][name].append((res.r, sum(t["pnl"] for t in res.trades)))
        totals["now (4h limit)"]["dd"].append(backtest.run(df4, p, start=start, feats=f4, costs=costs).max_dd)
        for kind, stop_kind in variants:
            tr, curve = simulate(df4, f4, d15, p, costs, kind, stop_kind, start_ts)
            s = summarize(tr, curve, cut)
            for name in ("research", "vault"):
                rs = [t[1] for t in tr if (t[0] < cut) == (name == "research")]
                totals[f"{kind}/{stop_kind}"][name].append((rs, s[name]["pnl"]))
            totals[f"{kind}/{stop_kind}"]["dd"].append(s["max_dd"])
        print(f"  {sym} done", flush=True)
    n_sym = len(totals["now (4h limit)"]["dd"])
    print(f"\n===== {mkt().name}: 4h direction + 15m entry ({n_sym} symbols, ~5 years; return = avg per symbol on "
          f"{START_EQ:.0f} start)")
    print(f"{'variant':18} | {'RESEARCH: trades  win  avgR  return':36} | {'VAULT: trades  win  avgR  return':33} | maxDD")
    for v, t in totals.items():
        row = []
        for name in ("research", "vault"):
            rs = np.concatenate([np.asarray(x[0], float) for x in t[name]]) if t[name] else np.array([])
            ret = np.mean([x[1] for x in t[name]]) / START_EQ if t[name] else 0.0
            row.append(f"{len(rs):6} {np.mean(rs > 0) if len(rs) else 0:4.0%} {rs.mean() if len(rs) else 0:+.3f} {ret:+7.1%}")
        print(f"{v:18} | {row[0]:36} | {row[1]:33} | {max(t['dd']) if t['dd'] else 0:.1%}")


if __name__ == "__main__":
    main()
