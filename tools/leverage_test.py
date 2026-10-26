"""What does a different leverage limit change? (owner, 2026-10-26)

The size of every trade is set by its risk (0.5% of the account if the stop is hit);
the leverage limit only caps how big all open positions together may be. So it
matters when the stop is close or several trades are open at once. This runs the
live settings with the limit at the current value and at 3x, 4x and 5x.

    python tools/leverage_test.py <market> <strategy_params.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[1]))

from core import improver, portfolio, risk_guard, significance, validation  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params  # noqa: E402


def main():
    p = clamp_params(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["params"])
    frames = improver.load_frames(Storage(), mkt().symbols)
    risk = mkt().cfg.setdefault("risk", {})
    now = risk.get("max_leverage", risk_guard.MAX_LEVERAGE)
    for lev in sorted({now, 3, 4, 5}):
        risk["max_leverage"] = lev
        wf, v = validation.walk_forward(frames, p), validation.vault(frames, p)
        r = portfolio.run(frames, p)                       # whole history, one shared account
        pv = portfolio.vault(frames, p)
        sg = significance.random_benchmark(frames, p, n=100)
        print(f"{mkt().name:11} 1:{lev:<2}{' (now)' if lev == now else '      '} | wf {wf['score']:+.2f} ({wf['positive_folds']}/4, "
              f"{wf['trades']} tr, {wf['avg_r']:+.3f}R) | vault {v['return']:+.2%} ({v['trades']} tr) | one account: "
              f"all years {r['return']:+.1%} dd {r['max_dd']:.1%}, vault {pv['return']:+.1%} dd {pv['max_dd']:.1%} "
              f"({pv['trades']} tr) | random p {sg['p_value']:.2f} | refused most: {list(r['refused'].items())[:2]}",
              flush=True)


if __name__ == "__main__":
    main()
