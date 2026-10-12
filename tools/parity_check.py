"""Parity check: v1 (crypto_ai) and v2 (trader_ai_v2) must give identical results
on the same database snapshot.

  python tools/parity_check.py <path to v1 crypto_ai> <snapshot bot.db>

Runs each version in its own Python process (with its own virtualenv) and
compares: per-coin backtest summaries, walk-forward, vault and the raw score /
signal series of the live settings.
"""
import json
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent

PROBE = r'''
import json, sys
import numpy as np
sys.path.insert(0, ROOT)
VERSION = "VER"
DB = r"SNAPSHOT"
if VERSION == "v1":
    from bot import backtest, library, market_data, validation
    from bot.features import add_static_features
    from bot.settings import SYMBOLS
    from bot.storage import Storage
    from bot.strategy import compute, load_params
    st = Storage(DB)
    load = lambda s: market_data.load_frame(st, s)
else:
    import markets
    from core.context import activate
    activate(markets.get("crypto"))
    from core import backtest, library, validation
    from core.context import mkt
    from core.features import add_static_features
    from core.storage import Storage
    from core.strategy import compute, load_params
    st = Storage(DB)
    SYMBOLS = mkt().symbols
    load = lambda s: mkt().load_frame(st, s)
p = load_params()
strats = library.approved()
frames = {s: add_static_features(load(s), strats) for s in SYMBOLS}
out = {"symbols": SYMBOLS, "approved": [s["name"] for s in strats],
       "walk_forward": validation.walk_forward(frames, p), "vault": validation.vault(frames, p), "per_symbol": {}}
for s, df in frames.items():
    f = compute(df, p)
    res = backtest.run(df, p)
    out["per_symbol"][s] = {"summary": res.summary(), "final_equity": float(res.equity[-1]),
                            "score_sum": float(np.nansum(f["score"].to_numpy())),
                            "signal": f["signal"].astype(int).tolist()}
print("JSON" + json.dumps(out))
'''


def run(version: str, root: Path, snapshot: str) -> dict:
    py = root / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = root / ".venv" / "bin" / "python"
    code = PROBE.replace("ROOT", repr(str(root))).replace("VER", version).replace("SNAPSHOT", snapshot)
    proc = subprocess.run([str(py), "-c", code], cwd=root, capture_output=True, text=True, encoding="utf-8")
    line = next((l for l in proc.stdout.splitlines() if l.startswith("JSON")), None)
    if line is None:
        raise SystemExit(f"{version} failed:\n{proc.stderr[-3000:]}")
    return json.loads(line[4:])


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    v1_root, snapshot = Path(sys.argv[1]), sys.argv[2]
    a, b = run("v1", v1_root, snapshot), run("v2", V2, snapshot)
    problems = []
    for key in ("symbols", "approved", "walk_forward", "vault"):
        if a[key] != b[key]:
            problems.append(f"{key}: v1={a[key]} v2={b[key]}")
    for s in a["per_symbol"]:
        x, y = a["per_symbol"][s], b["per_symbol"][s]
        if x["summary"] != y["summary"] or abs(x["final_equity"] - y["final_equity"]) > 1e-6:
            problems.append(f"{s} backtest: v1={x['summary']} v2={y['summary']}")
        diff = sum(i != j for i, j in zip(x["signal"], y["signal"]))
        if diff or len(x["signal"]) != len(y["signal"]):
            problems.append(f"{s}: {diff} signals differ (of {len(x['signal'])})")
        if abs(x["score_sum"] - y["score_sum"]) > 1e-6 * max(1.0, abs(x["score_sum"])):
            problems.append(f"{s}: score sum v1={x['score_sum']} v2={y['score_sum']}")
    print("walk-forward:", b["walk_forward"])
    print("vault:       ", b["vault"])
    for s, r in b["per_symbol"].items():
        print(f"{s:16} trades={r['summary']['trades']:4} return={r['summary']['return']:+.4f} "
              f"final={r['final_equity']:.4f} signals={len(r['signal'])}")
    if problems:
        print("\n❌ PARITY FAILED")
        print("\n".join(problems))
        raise SystemExit(1)
    print("\n✅ PARITY OK: v1 and v2 give identical results")


if __name__ == "__main__":
    main()
