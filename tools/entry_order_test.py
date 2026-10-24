"""Limit entry (wait for a small pull-back) vs market entry (take the signal at once).

The crypto bot sets a limit order entry_offset_atr away from the close and cancels
it after one candle; when the price runs away it never fills (ETH 2026-10-22..24).
This runs the same settings with both entry types and judges them the usual way
(market entries pay the taker fee and slippage instead of the maker fee).

    python tools/entry_order_test.py <market> <strategy_params.json>
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[1]))

from core import backtest, improver, portfolio, significance, strategy, validation  # noqa: E402
from core.storage import Storage  # noqa: E402


def judge(frames, p):
    wf, v = validation.walk_forward(frames, p), validation.vault(frames, p)
    pv, sg = portfolio.vault(frames, p), significance.random_benchmark(frames, p, n=100)
    tried = filled = 0
    for s, df in frames.items():
        res = backtest.run(df, p, start=validation.research_end(len(df)), feats=strategy.compute(df, p),
                           costs=mkt().costs_for(s))
        tried += len(res.trades) + res.missed
        filled += len(res.trades)
    return wf, v, pv, sg["p_value"], filled / tried if tried else 0.0


def main():
    raw = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    p = strategy.clamp_params(raw.get("params", raw))
    frames = improver.load_frames(Storage(), mkt().symbols)
    env_key = f"{mkt().name.upper()}_ENTRY_ORDER"
    for label, order in (("limit (now)", "maker"), ("market (no limit)", "market")):
        os.environ[env_key] = order
        wf, v, pv, pval, fill = judge(frames, p)
        print(f"{label:18} filled {fill:.0%} | wf {wf['score']:+.2f} ({wf['positive_folds']}/4, {wf['trades']} tr, "
              f"{wf['avg_r']:+.3f}R, {[f'{x:+.1%}' for x in wf['fold_returns']]}) | vault {v['return']:+.2%} "
              f"({v['trades']} tr, {v['avg_r']:+.3f}R, dd {v['max_dd']:.1%}) | one account {pv['return']:+.1%} "
              f"dd {pv['max_dd']:.1%} | random p {pval:.2f}", flush=True)
    os.environ.pop(env_key, None)


if __name__ == "__main__":
    main()
