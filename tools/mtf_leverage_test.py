"""4h direction + 15m entry with a tight stop (1R = the stop), at 1:100 and 1:1000 leverage.

Owner, 2026-10-26. Two ways to read "enter at 1:100 / 1:1000":

  risk   - the size still makes the stop cost exactly 0.5% of the account (1R); the
           leverage limit only allows the big sizes a very close stop needs.
           The hard 0.05% minimum stop distance stays (closer = noise).
  full   - the whole leverage is used: the position is 100x / 1000x the account,
           so a stop costs (stop distance x leverage) of the account.

Uses tools/mtf_test.py's simulator (15m rejection / break entries, stop just beyond
the pull-back, everything else as live). An account that falls to zero is "ruined"
and stops trading.

    python tools/mtf_leverage_test.py <market> <m15.db>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import mtf_test as T  # noqa: E402  (activates the market from argv[1])
from core import improver, risk_guard, validation  # noqa: E402
from core.context import mkt  # noqa: E402
from core.reward import max_drawdown  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import compute, load_params  # noqa: E402

real_check = risk_guard.check_entry
CASES = [("risk 0.5%, max 1:3 (now)", "risk", 3), ("risk 0.5%, max 1:100", "risk", 100),
         ("risk 0.5%, max 1:1000", "risk", 1000), ("FULL 1:100", "full", 100), ("FULL 1:1000", "full", 1000)]


def set_case(sizing, lev):
    cfg = mkt().cfg.setdefault("risk", {})
    cfg["max_leverage"] = lev
    cfg["min_stop_distance"] = risk_guard.ABS_MIN_STOP_DISTANCE
    risk_guard.ABS_MAX_LEVERAGE = max(lev, 10)
    if sizing == "risk":
        risk_guard.check_entry = real_check
        T.risk_guard.check_entry = real_check
        return

    def full(**kw):
        if kw["equity"] <= 0:
            return risk_guard.Decision(False, reason="ruined")
        return risk_guard.Decision(True, qty=lev * kw["equity"] / kw["entry"], reason="ok")
    risk_guard.check_entry = T.risk_guard.check_entry = full


def main():
    m15 = sys.argv[2]
    p = load_params()
    frames = improver.load_frames(Storage(), mkt().symbols)
    data = []
    for sym, df4 in frames.items():
        d15 = T.load15(m15, sym)
        if len(d15) > 1000:
            data.append((sym, df4, compute(df4, p), d15, int(df4.index[validation.research_end(len(df4))])))
    print(f"\n===== {mkt().name}: 4h direction, 15m entry, stop just beyond the pull-back = 1R "
          f"({len(data)} symbols, ~5 years, each starts with {T.START_EQ:.0f})")
    print(f"{'entry':9} {'sizing':26} | trades  win   avgR | avg return | worst symbol | max DD | ruined")
    for kind in ("rejection", "break"):
        for label, sizing, lev in CASES:
            set_case(sizing, lev)
            rets, dds, ruined, rs_all = [], [], 0, []
            for sym, df4, f4, d15, cut in data:
                start_ts = int(d15["ts"].iloc[0]) + 20 * 24 * 3_600_000
                tr, curve = T.simulate(df4, f4, d15, p, mkt().costs_for(sym), kind, "swing", start_ts)
                eq = T.START_EQ + np.cumsum([t[2] for t in tr]) if tr else np.array([T.START_EQ])
                if (eq <= 0).any():                 # ruined: stop at zero
                    ruined += 1
                    k = int(np.argmax(eq <= 0))
                    tr, eq = tr[:k + 1], np.minimum(eq[:k + 1], 0.0)
                rets.append(eq[-1] / T.START_EQ - 1)
                dds.append(max_drawdown(np.clip(np.concatenate([[T.START_EQ], eq]), 1e-9, None)))
                rs_all += [t[1] for t in tr]
            rs = np.array(rs_all)
            print(f"{kind:9} {label:26} | {len(rs):6} {np.mean(rs > 0):4.0%} {rs.mean():+.3f} | "
                  f"{np.mean(rets):+10.1%} | {min(rets):+12.1%} | {max(dds):6.1%} | {ruined}/{len(data)}", flush=True)
    risk_guard.check_entry = real_check


if __name__ == "__main__":
    main()
