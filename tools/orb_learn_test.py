"""Does learning from its own trades help? (owner, 2026-11-10)

The learner (core/orb_learn.py) starts in Oct 2021 knowing nothing (every arm at the
cost-adjusted prior) and walks forward through 5 years of 15m candles, deciding each
trade only from trades that had closed before it. Compared on the same candles with:
  london4   the fixed London bot now on brain 2 (pairs chosen in 2026 with hindsight)
  all_arms  every arm traded at full risk, no learning
  learner   arms weighted by what it learned, plus its lessons from bad trades
  beliefs   learner without the lessons (to see what each part adds)

    python tools/orb_learn_test.py
"""
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).resolve().parent.parent / "data" / "fx" / "bot.db"


def arm_trades(pair):
    from core import orb_learn
    db = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT ts, open, high, low, close FROM candles WHERE symbol=? AND tf='15m' ORDER BY ts",
                           db, params=(pair,))
    return [t for s in orb_learn.SESSIONS for t in orb_learn.candidates(df, pair, s)]


def summary(name, book, curve, start, end_ret, dd):
    taken = [t for t in book if t.get("weight", 1) > 0 and "pnl" in t]
    rs = np.array([t["r"] for t in taken]) if taken else np.array([0.0])
    eq = pd.Series([1 + v for _, v in curve], index=pd.to_datetime([t for t, _ in curve], unit="ms"))
    yearly = eq.resample("YE").last().pct_change().fillna(eq.resample("YE").last().iloc[0] - 1)
    last2 = eq[eq.index >= "2024-10-28"]
    l2 = last2.iloc[-1] / eq[eq.index < "2024-10-28"].iloc[-1] - 1 if len(last2) and (eq.index < "2024-10-28").any() else np.nan
    print(f"{name:9s} return {end_ret:+8.1%}  maxDD {dd:5.1%}  trades {len(taken):5d}  avg {rs.mean():+.3f}R  "
          f"last 2y {l2:+7.1%}  | by year " + " ".join(f"{d.year}:{v:+.0%}" for d, v in yearly.items()))


def main():
    from core import orb15, orb_learn
    # compare like for like over all 5 years: 0.5% risk and no drawdown stop (at 2% the -45% stop
    # halts every variant somewhere in 2022-23 and the comparison ends there)
    orb15.RISK, orb15.MAX_DD = 0.005, 0.99
    with ProcessPoolExecutor(10) as ex:
        cands = [t for ts in ex.map(arm_trades, list(orb_learn.PAIRS)) for t in ts]
    start = min(t["entry_ts"] for t in cands) + 30 * 86_400_000     # indicators warm up first
    print(f"{len(cands)} candidate trades in {len({t['arm'] for t in cands})} arms, from "
          f"{pd.to_datetime(start, unit='ms'):%Y-%m-%d}\n")

    lon4 = [t for t in cands if t["session"] == "london" and t["pair"] in orb15.PAIRS]
    acc = orb15.account(lon4, start)
    summary("london4", [{**t, "weight": 1} for t in acc["trades"]], acc["curve"], start, acc["return"], acc["max_dd"])

    keep = (orb_learn.FULL_P, orb_learn.HALF_P, orb_learn.LESSON_P)
    orb_learn.FULL_P = orb_learn.HALF_P = -1.0                       # every arm always full risk
    orb_learn.LESSON_P = 2.0                                         # no lessons
    r = orb_learn.replay(cands, start)
    summary("all_arms", r["book"], r["curve"], start, r["return"], r["max_dd"])

    orb_learn.FULL_P, orb_learn.HALF_P = keep[0], keep[1]
    r = orb_learn.replay(cands, start)
    summary("beliefs", r["book"], r["curve"], start, r["return"], r["max_dd"])

    orb_learn.LESSON_P = keep[2]
    r = orb_learn.replay(cands, start)
    summary("learner", r["book"], r["curve"], start, r["return"], r["max_dd"])

    print("\nwhat the learner believes at the end (avg R, P(profitable)) - best and worst arms:")
    b = sorted(r["beliefs"].items(), key=lambda x: -x[1][0])
    for a, (m, p) in b[:6] + [("...", (np.nan, np.nan))] + b[-4:]:
        print(f"   {a:18s} {m:+.3f}R  P={p:.2f}")
    print("lessons it holds at the end:", r["rules"])
    risked = [t for t in r["book"] if t["weight"] > 0]
    print(f"traded with risk: {len(risked)} of {len(r['book'])} candidate trades; "
          f"blocked by lessons: {sum(1 for t in r['book'] if t['blocked_by'])}")


def grid():
    """How cautious should the learner be? Chosen on 2021-11..2024-10, judged on the 2 years after."""
    import itertools
    from core import orb15, orb_learn
    orb15.RISK, orb15.MAX_DD = 0.005, 0.99
    with ProcessPoolExecutor(10) as ex:
        cands = [t for ts in ex.map(arm_trades, list(orb_learn.PAIRS)) for t in ts]
    start = min(t["entry_ts"] for t in cands) + 30 * 86_400_000
    cut = int(pd.Timestamp("2024-10-28", tz="UTC").timestamp() * 1000)
    print(f"{'prior_n':>7s} {'full_p':>6s} {'half_p':>6s} | {'choose: R  trades':>18s} | {'judge: R  trades':>17s}")
    for prior_n, full_p, half in itertools.product((20, 50, 100), (0.75, 0.85, 0.95), (True, False)):
        orb_learn.PRIOR_N, orb_learn.FULL_P = prior_n, full_p
        orb_learn.HALF_P = 0.55 if half else full_p
        r = orb_learn.replay(cands, start)
        risked = [t for t in r["book"] if t["weight"] > 0]
        a = [t["r"] * t["weight"] for t in risked if t["entry_ts"] < cut]
        b = [t["r"] * t["weight"] for t in risked if t["entry_ts"] >= cut]
        print(f"{prior_n:7d} {full_p:6.2f} {orb_learn.HALF_P:6.2f} | {sum(a):+9.1f}R {len(a):6d} | {sum(b):+8.1f}R {len(b):6d}")


def snapshot():
    """What the learner knows after all the history on this machine -> data/fx/orb_learn_snapshot.json
    (every closed candidate trade with its situation; brain 2 carries on from it)."""
    import json
    from core import orb_learn
    with ProcessPoolExecutor(10) as ex:
        cands = [t for ts in ex.map(arm_trades, list(orb_learn.PAIRS)) for t in ts]
    end = max(t["exit_ts"] for t in cands)
    rows = [{k: (round(t[k], 4) if isinstance(t[k], float) else t[k]) for k in orb_learn.SNAP_FIELDS} for t in cands]
    out = DB.parent / "orb_learn_snapshot.json"
    out.write_text(json.dumps({"end": end, "trades": rows}), encoding="utf-8")
    print(f"{len(rows)} trades up to {pd.to_datetime(end, unit='ms')} -> {out}")


if __name__ == "__main__":
    grid() if "--grid" in sys.argv else snapshot() if "--snapshot" in sys.argv else main()
