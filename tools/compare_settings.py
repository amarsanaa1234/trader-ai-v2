"""Judge saved settings with today's strict checks on any timeframe.

    CRYPTO_TIMEFRAME=1h python tools/compare_settings.py crypto <strategy_params.json> [label]

Walk-forward (4 windows), vault (newest 15%), shared-account vault, random-entry
benchmark (the market's own validation rules decide what counts as passing).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402

activate(markets.get(sys.argv[1]))

from core import improver, portfolio, significance, validation  # noqa: E402
from core.storage import Storage  # noqa: E402
from core.strategy import clamp_params  # noqa: E402


def main():
    p = clamp_params(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["params"])
    label = sys.argv[3] if len(sys.argv) > 3 else Path(sys.argv[2]).stem
    frames = improver.load_frames(Storage(), mkt().symbols)
    days = max((df.index[-1] - df.index[0]) / 86_400_000 for df in frames.values())
    wf, v = validation.walk_forward(frames, p), validation.vault(frames, p)
    pv, sg = portfolio.vault(frames, p), significance.random_benchmark(frames, p, n=100)
    gate_wf = wf["positive_folds"] >= improver.MIN_POSITIVE_FOLDS
    gate_v = v["return"] > 0 and v["trades"] >= improver.min_vault_trades()
    gate_p = sg["p_value"] <= float((mkt().cfg.get("validation") or {}).get("max_p_value", 1.0))
    print(f"{label:34} [{mkt().timeframe}, thr {p['entry_threshold']:.2f}] wf {wf['score']:+.2f} ({wf['positive_folds']}/4, "
          f"{wf['trades']} tr = {wf['trades'] / (days * 0.85 / 30):.0f}/month, {wf['avg_r']:+.3f}R, "
          f"{' '.join(f'{x:+.1%}' for x in wf['fold_returns'])}) | vault {v['return']:+.2%} ({v['trades']} tr, {v['avg_r']:+.3f}R, "
          f"dd {v['max_dd']:.1%}) | one account {pv['return']:+.1%} dd {pv['max_dd']:.1%} | random p {sg['p_value']:.2f} | "
          f"passes: walk-forward {'✅' if gate_wf else '❌'} vault {'✅' if gate_v else '❌'} random {'✅' if gate_p else '❌'}",
          flush=True)


if __name__ == "__main__":
    main()
