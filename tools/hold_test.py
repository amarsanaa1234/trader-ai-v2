"""How long may a trade stay open? Max hold of 3..21 days at a chosen risk per trade.

The short-term rule caps holds at 7 days (core/strategy.py MAX_HOLD_DAYS); this
lifts the cap for the test only and sets max_hold_bars to each value.

    python tools/hold_test.py <market> <strategy_params.json> [risk, e.g. 0.01]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[1]))

from core import improver, portfolio, significance, strategy, validation  # noqa: E402
from core.storage import Storage  # noqa: E402


def main():
    risk = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    mkt().cfg.setdefault("risk", {})["max_risk_per_trade"] = risk
    strategy.MAX_HOLD_DAYS = 365                        # test only: lift the 7-day rule
    base = strategy.clamp_params(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["params"])
    frames = improver.load_frames(Storage(), mkt().symbols)
    per_day = 86_400_000 // mkt().tf_ms()
    print(f"{mkt().name}: risk {risk:.1%} per trade, live max hold {base['max_hold_bars']} candles "
          f"({base['max_hold_bars'] / per_day:.1f} days)")
    for days in (3, 5, 7, 10, 14, 21):
        p = {**base, "max_hold_bars": days * per_day}
        wf, v = validation.walk_forward(frames, p), validation.vault(frames, p)
        res = portfolio.run(frames, p, with_curve=True)
        pv = portfolio.vault(frames, p)
        sg = significance.random_benchmark(frames, p, n=100)
        eq = pd.Series(res["curve"], index=pd.to_datetime(res["times"], unit="ms", utc=True))
        monthly = eq.resample("ME").last().pct_change().dropna()
        print(f"  max {days:2} days | wf {wf['score']:+.2f} ({wf['positive_folds']}/4, {wf['avg_r']:+.3f}R) | "
              f"vault {v['return']:+.2%} ({v['trades']} tr, {v['avg_r']:+.3f}R) | one account: all years "
              f"{res['return']:+.1%} dd {res['max_dd']:.1%}, worst month {monthly.min():+.1%}, vault {pv['return']:+.1%} "
              f"dd {pv['max_dd']:.1%} | random p {sg['p_value']:.2f}", flush=True)


if __name__ == "__main__":
    main()
