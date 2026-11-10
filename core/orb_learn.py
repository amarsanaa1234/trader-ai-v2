"""Brain 2 · 15m FX learner: session breakouts that learn from their own trades (owner, 2026-11-10).

The owner wants the bot to improve by itself from its good and bad trades, not by
swapping indicator settings. So the trading rules here are FIXED in advance and
never tuned; what changes is what the bot has learned from the trades it saw:

1. Arms. One opening-range breakout per session (Tokyo, London, New York) on each
   of the 10 pairs = 30 arms, each with the same rule shape as the London bot
   (core/orb15.py): range, first close outside it in the entry window with the 1h
   trend, 2xATR stop, no take-profit, flat at the session's end.
2. Belief per arm. Every closed trade updates a Bayesian estimate of that arm's
   average R (normal model, sd 1.2R, prior -0.05R worth PRIOR_N trades: costs make
   a new arm guilty until it proves itself; old trades fade with a one-year
   half-life because markets change). The arm trades with full risk only when it
   is probably profitable (P(avg R > 0) >= 0.75); otherwise only on paper inside
   the paper bot - its trades are still recorded and still teach it, so an arm
   can earn its way back.
3. Lessons from bad trades. Every trade stores its situation: range width,
   strength of the 1h trend, volatility vs normal, weekday. Once a week the bot
   looks across all arms for situations that clearly lose (P(avg R < 0) >= 0.9
   over at least MIN_LESSON trades and 0.15R worse than overall) and stops
   trading them; a lesson the newer trades no longer support is dropped again.

Learning only ever uses trades that had closed before the decision. Paper only.
State: "brain2_learn" in the FX database.
"""
import math

import numpy as np
import pandas as pd

from . import orb15

PAIRS = {"EUR_USD": 0.0001382, "USD_JPY": 0.0000967, "GBP_USD": 0.0001361, "AUD_USD": 0.0001861,
         "USD_CAD": 0.0001271, "USD_CHF": 0.0001836, "NZD_USD": 0.0002555, "EUR_JPY": 0.0001454,
         "GBP_JPY": 0.0001508, "EUR_GBP": 0.0001534}
# local clock of each session: range (r0, r1], entry (r1, e1], flat from `flat` (minutes after midnight)
SESSIONS = {
    "tokyo": {"tz": "Asia/Tokyo", "r0": 420, "r1": 540, "e1": 660, "flat": 900},          # 07-09 / 09-11 / 15:00
    "london": {"tz": "Europe/London", "r0": 0, "r1": 480, "e1": 600, "flat": 960},         # 00-08 / 08-10 / 16:00
    "ny": {"tz": "America/New_York", "r0": 480, "r1": 570, "e1": 690, "flat": 960},       # 08-09:30 / -11:30 / 16:00
}
SD_R, PRIOR_MEAN, PRIOR_N, HALF_LIFE_DAYS = 1.2, -0.05, 50, 365
# How cautious: chosen on 2021-11..2024-10, judged on the 2 years after (tools/orb_learn_test.py --grid).
# Half risk for "leaning profitable" arms lost in every setting there, so an arm trades only once it
# is probably profitable: 50/0.75 -> +45.6R choosing, +19.3R (172 trades) on the unseen 2 years.
FULL_P = HALF_P = 0.75
LESSON_P, MIN_LESSON, LESSON_GAP, LESSON_EVERY_MS = 0.90, 30, 0.15, 7 * 86_400_000
DAY_MS, KEY = 86_400_000, "brain2_learn"
BUCKETS = {"width": (0.8, 1.6), "trend": (0.3, 1.0), "vol": (0.8, 1.25)}


SNAP_FIELDS = ("arm", "pair", "session", "entry_ts", "exit_ts", "r", "width", "trend", "vol", "dow")


def snapshot_path():
    from .context import mkt
    return mkt().data_dir / "orb_learn_snapshot.json"


def step(st, now: int) -> dict | None:
    """Brain 2: carry the learner on to the last closed 15m candle. It starts from what it learned
    on 5 years of history (snapshot made by tools/orb_learn_test.py --snapshot, ending where that
    history ended) and from then on learns from the new trades as they close."""
    import json
    path = snapshot_path()
    if not path.exists():
        return None
    snap = json.loads(path.read_text(encoding="utf-8"))
    known, snap_end = snap["trades"], int(snap["end"])
    b = st.get(KEY) or {"start": snap_end, "live_from": now}
    cands = []
    for pair in PAIRS:
        df = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf=? AND ts>=? "
                               "ORDER BY ts", st.db,
                               params=(pair, orb15.TF, b["start"] - orb15.WARMUP_DAYS * DAY_MS))
        if len(df) < 96 * 25:
            continue
        cands += [t for s in SESSIONS for t in candidates(df, pair, s) if t["entry_ts"] > snap_end]
    r = replay(cands, b["start"], known)
    risked = [t for t in r["book"] if t["weight"] > 0 and "pnl" in t]
    live = [t for t in risked if t["entry_ts"] >= b["live_from"]]
    beliefs = sorted(r["beliefs"].items(), key=lambda x: -x[1][0])
    b.update(equity=r["equity"], **{"return": r["return"]}, max_dd=r["max_dd"], n_trades=len(risked),
             avg_r=round(float(np.mean([t["r"] for t in risked])), 3) if risked else 0.0,
             paper_only=sum(1 for t in r["book"] if t["weight"] == 0 and not t["blocked_by"]),
             blocked=sum(1 for t in r["book"] if t["blocked_by"]),
             live_trades=len(live), live_r=round(float(sum(t["r"] for t in live)), 2),
             learned_from=len(r["known"]), curve=r["curve"][-2000:], updated=now,
             trusted=[{"arm": a, "avg_r": round(m, 3), "p": round(p, 2)} for a, (m, p) in beliefs if p >= FULL_P],
             worst=[{"arm": a, "avg_r": round(m, 3), "p": round(p, 2)} for a, (m, p) in beliefs[-3:]],
             lessons=[{"tag": k, "n": n, "avg_r": m} for k, (n, m) in r["rules"].items()],
             trades=[{"symbol": t["pair"], "session": t["session"], "ts": t["exit_ts"],
                      "side": "long" if t["side"] > 0 else "short", "r": round(t["r"], 2),
                      "pnl": t.get("pnl", 0.0), "weight": t["weight"], "p": t["p_pos"],
                      "blocked_by": t["blocked_by"], "live": t["entry_ts"] >= b["live_from"]}
                     for t in r["book"] if "pnl" in t or t["weight"] == 0][-60:])
    st.set(KEY, b)
    return b


def candidates(df_raw: pd.DataFrame, pair: str, session: str) -> list[dict]:
    """Every breakout trade the arm's fixed rule would take, with the situation it started in."""
    s = SESSIONS[session]
    df = orb15.features(df_raw)
    t = pd.to_datetime(df.ts, unit="ms", utc=True).dt.tz_convert(s["tz"])
    lc = t + pd.Timedelta(minutes=15)
    mins = (lc.dt.hour * 60 + lc.dt.minute).to_numpy()
    day = (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day).to_numpy()
    c, a, ad = df.close.to_numpy(), df.atr.to_numpy(), df.atr_d.to_numpy()
    rng = (mins > s["r0"]) & (mins <= s["r1"])
    g = pd.DataFrame({"d": day, "h": np.where(rng, df.high, np.nan), "l": np.where(rng, df.low, np.nan)}).groupby("d")
    rh, rl = g.h.transform("max").to_numpy(), g.l.transform("min").to_numpy()
    width = (rh - rl) / ad
    ok = (mins > s["r1"]) & (mins <= s["e1"]) & (width <= 6.0)
    up, dn = ok & (c > rh), ok & (c < rl)
    first = (up | dn) & (pd.Series((up | dn).astype(int)).groupby(day).cumsum().to_numpy() == 1)
    f, sl_, hc = df.h1_f.to_numpy(), df.h1_s.to_numpy(), df.h1_c.to_numpy()
    up &= (f > sl_) & (hc > f)
    dn &= (f < sl_) & (hc < f)
    d = np.zeros(len(df), np.int64)
    d[first & up], d[first & dn] = 1, -1
    trades, _ = orb15.simulate(df, {"dir": d, "sl": a * orb15.P["k"], "x": mins >= s["flat"]}, PAIRS[pair])
    ts = df.ts.to_numpy()
    out = []
    for tr in trades:
        i = int(np.searchsorted(ts, tr["entry_ts"])) - 1        # the decision candle
        out.append({**tr, "pair": pair, "session": session, "arm": f"{pair}:{session}",
                    "width": float(width[i]), "trend": float(abs(f[i] - sl_[i]) / a[i]), "vol": float(a[i] / ad[i]),
                    "dow": int(t.iloc[i].day_of_week)})
    return out


def bucket(tr: dict) -> list[str]:
    tags = [f"dow={tr['dow']}"]
    for k, (lo, hi) in BUCKETS.items():
        tags.append(f"{k}={'low' if tr[k] < lo else 'high' if tr[k] > hi else 'mid'}")
    return tags


def _post(rs, ages_days):
    """Posterior mean and P(mean > 0) of an arm's average R from its decayed past trades."""
    w = np.power(0.5, np.asarray(ages_days) / HALF_LIFE_DAYS) if len(rs) else np.array([])
    n = PRIOR_N + w.sum()
    mean = (PRIOR_N * PRIOR_MEAN + float(np.dot(w, rs))) / n
    return mean, 0.5 * (1 + math.erf(mean / (SD_R / math.sqrt(n)) / math.sqrt(2)))


def lessons(closed: list[dict]) -> dict:
    """Situations that clearly lose across all arms: {tag: (n, avg R)}."""
    if len(closed) < MIN_LESSON:
        return {}
    allr = np.array([t["r"] for t in closed])
    base, out = allr.mean(), {}
    by = {}
    for t in closed:
        for tag in bucket(t):
            by.setdefault(tag, []).append(t["r"])
    for tag, rs in by.items():
        rs = np.asarray(rs)
        if len(rs) < MIN_LESSON:
            continue
        m, se = rs.mean(), SD_R / math.sqrt(len(rs))
        p_neg = 0.5 * (1 + math.erf((-m / se) / math.sqrt(2)))
        if p_neg >= LESSON_P and m < base - LESSON_GAP:
            out[tag] = (len(rs), round(float(m), 3))
    return out


def replay(cands: list[dict], start: int, snapshot: list | None = None) -> dict:
    """Walk through the candidate trades in time order. Each decision uses only trades closed
    before it. `snapshot`: earlier closed trades the learner already knows (same fields)."""
    cands = sorted(cands, key=lambda t: t["entry_ts"])
    known = list(snapshot or [])            # closed trades it has learned from
    by_arm = {}
    for t in known:
        by_arm.setdefault(t["arm"], []).append(t)
    pending = []                            # taken (or paper) trades not closed yet
    rules, rules_at = lessons(known), start
    eq = peak = orb15.START_EQUITY
    day, day_start, max_dd = None, eq, 0.0
    curve, book, open_risk = [[start, 0.0]], [], {}
    for k, tr in enumerate(cands):
        now = tr["entry_ts"]
        # trades that closed before now: book their P&L and learn from them
        done = sorted([p for p in pending if p["exit_ts"] <= now], key=lambda p: p["exit_ts"])
        for p in done:
            pending.remove(p)
            known.append(p)
            by_arm.setdefault(p["arm"], []).append(p)
            if p["k"] in open_risk:
                pnl = p["r"] * open_risk.pop(p["k"])
                eq += pnl
                peak = max(peak, eq)
                max_dd = max(max_dd, 1 - eq / peak)
                p["pnl"] = round(pnl, 2)
                curve.append([int(p["exit_ts"]), round(eq / orb15.START_EQUITY - 1, 5)])
        if now - rules_at >= LESSON_EVERY_MS:
            rules, rules_at = lessons(known), now
        if any(p["arm"] == tr["arm"] for p in pending):
            continue                        # one position per arm
        if now // DAY_MS != day:
            day, day_start = now // DAY_MS, eq
        mine = by_arm.get(tr["arm"], [])
        mean, p_pos = _post([t["r"] for t in mine], [(now - t["exit_ts"]) / DAY_MS for t in mine])
        hit = [tag for tag in bucket(tr) if tag in rules]
        weight = 0.0 if hit else 1.0 if p_pos >= FULL_P else 0.5 if p_pos >= HALF_P else 0.0
        if eq / day_start - 1 <= -orb15.DAILY_LOSS or eq / peak - 1 <= -orb15.MAX_DD:
            weight = 0.0
        rec = {**tr, "k": k, "weight": weight, "belief": round(mean, 3), "p_pos": round(p_pos, 2),
               "blocked_by": hit}
        if tr["entry_ts"] >= start and weight > 0:
            open_risk[k] = orb15.RISK * weight * eq
        pending.append(rec)
        if tr["entry_ts"] >= start:
            book.append(rec)
    for p in sorted(pending, key=lambda p: p["exit_ts"]):   # whatever closed by the last candle
        known.append(p)
        if p["k"] in open_risk:
            pnl = p["r"] * open_risk.pop(p["k"])
            eq += pnl
            peak = max(peak, eq)
            max_dd = max(max_dd, 1 - eq / peak)
            p["pnl"] = round(pnl, 2)
            curve.append([int(p["exit_ts"]), round(eq / orb15.START_EQUITY - 1, 5)])
    arms = {}
    for t in known:
        arms.setdefault(t["arm"], []).append(t)
    end = max([t["exit_ts"] for t in known], default=start)
    beliefs = {a: _post([t["r"] for t in ts], [(end - t["exit_ts"]) / DAY_MS for t in ts]) for a, ts in arms.items()}
    return {"equity": round(eq, 2), "return": round(eq / orb15.START_EQUITY - 1, 5), "max_dd": round(max_dd, 4),
            "book": book, "curve": curve, "known": known, "rules": lessons(known), "beliefs": beliefs}
