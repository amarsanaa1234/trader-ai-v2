"""0.5% vs 1% risk per trade on the live settings (owner, 2026-10-26).

Trades and their R stay the same; the money per trade doubles, and so do the
swings, which can trip the hard limits (daily -5% kill switch, -10% / -20% drawdown
steps that cut the risk, -30% halt). One shared account per market, all years and
the vault.

    python tools/risk_test.py <market> <strategy_params.json>
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

from core import improver, portfolio, validation  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params  # noqa: E402


def main():
    p = clamp_params(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["params"])
    frames = improver.load_frames(Storage(), mkt().symbols)
    risk = mkt().cfg.setdefault("risk", {})
    for r in (0.005, 0.01):
        risk["max_risk_per_trade"] = r
        res = portfolio.run(frames, p, with_curve=True)
        eq = pd.Series(res["curve"], index=pd.to_datetime(res["times"], unit="ms", utc=True))
        monthly = eq.resample("ME").last().pct_change().dropna()
        years = eq.resample("YE").last().pct_change().dropna()
        pv = portfolio.vault(frames, p)
        v = validation.vault(frames, p)
        print(f"{mkt().name:11} risk {r:.1%} | one account all years {res['return']:+7.1%} (max dd {res['max_dd']:5.1%}, "
              f"worst month {monthly.min():+.1%}, losing months {np.mean(monthly < 0):.0%}) | vault {pv['return']:+.1%} "
              f"dd {pv['max_dd']:.1%} | per-symbol vault {v['return']:+.2%} | {res['trades']} trades | "
              f"by year {' '.join(f'{d.year}:{x:+.0%}' for d, x in years.items())}", flush=True)


if __name__ == "__main__":
    main()
