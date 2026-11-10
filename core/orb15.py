"""Brain 2 · 15-minute FX: London opening-range breakout, paper only (owner, 2026-11-10).

The owner asked for a profitable 15m FX bot on brain 2 to watch for a few days.
It is the strategy from the 15m research (Downloads/fx15_mt5/docs/RESEARCH.md,
2026-11-01) taken over unchanged - nine strategy families and 1,110 settings were
tested, every other one lost after OANDA spreads; this one was positive on two
unseen years (valid +0.060R, holdout +0.079R per trade). The holdout was opened
once then, so the rules are NOT re-tuned here: tuning again on the same years
would only flatter them. A thin edge, not a promise.

Rules (decided at the close of each 15m candle, London time):
  range   high/low of 00:00-08:00
  entry   the day's first close outside the range between 08:00 and 10:00, only if
          the 1h trend agrees (EMA50 vs EMA200 and the close beyond EMA50);
          market order at the next candle's open, paying the spread
  stop    2 x ATR(14, 15m); no take-profit; closed at 16:00, after 64 candles,
          or Friday 20:00 UTC (never over the weekend)
  pairs   USD_JPY, EUR_JPY, GBP_JPY, GBP_USD (London spread < 20% of a 15m ATR)
  risk    2% of the paper account per trade (owner's choice for this bot on 11/01),
          no new entry after -6% in a UTC day or -45% from the peak
It never places an order. State: "brain2_orb15" in the FX database.
"""
import numpy as np
import pandas as pd

PAIRS = {"USD_JPY": 0.0000967, "EUR_JPY": 0.0001454, "GBP_JPY": 0.0001508, "GBP_USD": 0.0001361}
P = {"r0": 0, "r1": 480, "e1": 600, "k": 2.0, "hold": 64, "flat": 960, "wmin": 0.0, "wmax": 6.0}
TF, BAR_MS, DAY_MS = "15m", 900_000, 86_400_000
RISK, DAILY_LOSS, MAX_DD = 0.02, 0.06, 0.45
SLIP = 0.3                      # extra share of a spread paid on stop exits
START_EQUITY, START_DAYS_BACK, WARMUP_DAYS = 10_000.0, 30, 60
KEY = "brain2_orb15"


def _ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Only the columns the ORB needs, computed exactly as in the research (fx15/features.py)."""
    df = df.copy().reset_index(drop=True)
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    t = pd.to_datetime(df.ts, unit="ms", utc=True)
    lc = t.dt.tz_convert("Europe/London") + pd.Timedelta(minutes=15)   # London time at the candle's close
    df["utc_h"], df["dow"] = t.dt.hour, t.dt.day_of_week
    df["ldn_min"] = (lc.dt.hour * 60 + lc.dt.minute).to_numpy()
    ldn = t.dt.tz_convert("Europe/London")
    df["ldn_day"] = (ldn.dt.year * 10000 + ldn.dt.month * 100 + ldn.dt.day).to_numpy()
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    df["atr"] = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()
    df["atr_d"] = pd.Series(df.atr).rolling(96 * 20, min_periods=96).mean().to_numpy()
    s = pd.Series(c, index=t)
    hc = s.resample("1h", label="left", closed="left").last().dropna()   # 1h candles, known once closed
    htf = pd.DataFrame({"f": _ema(hc.to_numpy(), 50), "s": _ema(hc.to_numpy(), 200), "c": hc.to_numpy()},
                       index=hc.index + pd.Timedelta("1h"))
    m = pd.merge_asof(pd.DataFrame({"t": t + pd.Timedelta(minutes=15)}), htf.rename_axis("t").reset_index(),
                      on="t", direction="backward")
    df["h1_f"], df["h1_s"], df["h1_c"] = m.f.to_numpy(), m.s.to_numpy(), m.c.to_numpy()
    return df


def signals(df: pd.DataFrame) -> dict:
    """fx15/strategies.orb with the chosen settings (sl=atr, tp=0, htf=h1)."""
    n = len(df)
    day, mins = df.ldn_day.to_numpy(), df.ldn_min.to_numpy()
    c, a = df.close.to_numpy(), df.atr.to_numpy()
    rng = (mins > P["r0"]) & (mins <= P["r1"])
    g = pd.DataFrame({"d": day, "h": np.where(rng, df.high, np.nan), "l": np.where(rng, df.low, np.nan)}).groupby("d")
    rh, rl = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()
    width = (rh - rl) / df.atr_d.to_numpy()
    ok = (mins > P["r1"]) & (mins <= P["e1"]) & (width >= P["wmin"]) & (width <= P["wmax"])
    up, dn = ok & (c > rh), ok & (c < rl)
    hits = (up | dn).astype(int)
    first = (up | dn) & (pd.Series(hits).groupby(day).cumsum().to_numpy() == 1)
    f, s, hc = df.h1_f.to_numpy(), df.h1_s.to_numpy(), df.h1_c.to_numpy()
    up &= (f > s) & (hc > f)
    dn &= (f < s) & (hc < f)
    d = np.zeros(n, np.int64)
    d[first & up], d[first & dn] = 1, -1
    flat = mins >= P["flat"]
    return {"dir": d, "sl": a * P["k"], "x": flat}


def simulate(df: pd.DataFrame, sig: dict, spread_rel: float):
    """fx15/engine.simulate: entry at the next open, stop checked inside each later candle
    (gaps fill at the open), exit at the 16:00 / 64-candle / Friday close. Returns the closed
    trades and the open one (if any) at the last candle."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    uh, dow, ts = df.utc_h.to_numpy(), df.dow.to_numpy(), df.ts.to_numpy()
    mult = np.where(uh == 21, 3.0, np.where(uh == 22, 4.0, np.where(uh == 23, 1.5, 1.0)))
    spread = spread_rel * mult
    no_entry = np.isin(uh, (21, 22, 23)) | ((dow == 4) & (uh >= 19)) | (dow >= 5)
    force_flat = (dow == 4) & (uh >= 20)
    d_arr, sl, x = sig["dir"], sig["sl"], sig["x"]
    n, i, trades, open_pos = len(c), 0, [], None
    while i < n - 2:
        d = int(d_arr[i])
        if d == 0 or no_entry[i] or not (sl[i] > 0):
            i += 1
            continue
        e, s = o[i + 1], sl[i]
        cost, stop = spread[i + 1] * e, e - d * s
        j, exit_px, stopped = i + 1, None, False
        while j < n:
            if j > i + 1 and ((d == 1 and o[j] <= stop) or (d == -1 and o[j] >= stop)):
                exit_px, stopped = o[j], True
                break
            if (d == 1 and l[j] <= stop) or (d == -1 and h[j] >= stop):
                exit_px, stopped = stop, True
                break
            if x[j] or force_flat[j] or j - i >= P["hold"]:
                exit_px = c[j]
                break
            j += 1
        if exit_px is None:            # still open at the last candle
            open_pos = {"side": d, "entry": float(e), "stop": float(stop), "dist": float(s),
                        "opened": int(ts[i + 1]), "price": float(c[-1]),
                        "r_now": round(float((d * (c[-1] - e) - cost) / s), 2)}
            break
        if stopped:
            cost += SLIP * spread[j] * e
        trades.append({"entry_ts": int(ts[i + 1]), "exit_ts": int(ts[j]) + BAR_MS, "side": d,
                       "entry": float(e), "exit": float(exit_px),
                       "r": float((d * (exit_px - e) - cost) / s), "reason": "stop" if stopped else "time"})
        i = j
    return trades, open_pos


def account(trades: list, start: int) -> dict:
    """One paper account over all pairs: 2% risk per trade at entry, P&L booked at exit,
    no new entry after -6% in a UTC day or -45% from the peak."""
    events = sorted([(t["entry_ts"], 1, k) for k, t in enumerate(trades) if t["entry_ts"] >= start] +
                    [(t["exit_ts"], 0, k) for k, t in enumerate(trades) if t["entry_ts"] >= start])
    eq = peak = START_EQUITY
    day, day_start, risk_of, taken, curve, max_dd = None, eq, {}, [], [[start, 0.0]], 0.0
    for ts, is_entry, k in events:
        d = ts // DAY_MS
        if d != day:
            day, day_start = d, eq
        if is_entry:
            if eq / day_start - 1 > -DAILY_LOSS and eq / peak - 1 > -MAX_DD:
                risk_of[k] = RISK * eq
        elif k in risk_of:
            pnl = trades[k]["r"] * risk_of.pop(k)
            eq += pnl
            peak = max(peak, eq)
            max_dd = max(max_dd, 1 - eq / peak)
            taken.append({**trades[k], "pnl": round(pnl, 2)})
            curve.append([int(ts), round(eq / START_EQUITY - 1, 5)])
    return {"equity": round(eq, 2), "return": round(eq / START_EQUITY - 1, 5), "max_dd": round(max_dd, 4),
            "trades": taken, "curve": curve, "open_risk": {k: v for k, v in risk_of.items()}}


def step(st, now: int) -> dict:
    """Recompute the paper bot from its start on the stored 15m candles (deterministic, so a
    closed trade never changes) and save it for the dashboard."""
    b = st.get(KEY) or {"start": (now // DAY_MS - START_DAYS_BACK) * DAY_MS, "live_from": now}
    trades, opens = [], []
    for pair, spread in PAIRS.items():
        df = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf=? AND ts>=? "
                               "ORDER BY ts", st.db, params=(pair, TF, b["start"] - WARMUP_DAYS * DAY_MS))
        if len(df) < 96 * 25:
            continue
        df = features(df)
        tr, op = simulate(df, signals(df), spread)
        trades += [{**t, "symbol": pair} for t in tr]
        if op and op["opened"] >= b["start"]:
            opens.append({**op, "symbol": pair, "side": "long" if op["side"] > 0 else "short"})
    acc = account(trades, b["start"])
    rs = [t["r"] for t in acc["trades"]]
    live = [t for t in acc["trades"] if t["entry_ts"] >= b["live_from"]]
    b.update(equity=acc["equity"], **{"return": acc["return"]}, max_dd=acc["max_dd"], n_trades=len(rs),
             avg_r=round(float(np.mean(rs)), 3) if rs else 0.0,
             win=round(float(np.mean([r > 0 for r in rs])), 3) if rs else None,
             live_trades=len(live), live_r=round(float(np.sum([t["r"] for t in live])), 2),
             curve=acc["curve"][-2000:], open=opens, updated=now,
             trades=[{"symbol": t["symbol"], "ts": t["exit_ts"], "side": "long" if t["side"] > 0 else "short",
                      "r": round(t["r"], 2), "pnl": t["pnl"], "reason": t["reason"],
                      "live": t["entry_ts"] >= b["live_from"]} for t in acc["trades"]][-300:])
    st.set(KEY, b)
    return b
