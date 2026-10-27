"""Brain 2 (core/brain2.py): a paper bot, carried on hour by hour, never trading."""
from pathlib import Path

from conftest import synthetic_frame
from core import portfolio
from core.strategy import default_params

FRESH = {"equity": 10_000.0, "peak": 10_000.0, "day": None, "day_start": 10_000.0, "halted": False,
         "trades_today": 0, "last_close": {}, "pos": {}, "pend": {}}


def test_carrying_on_step_by_step_gives_the_same_trades_as_one_run():
    frames = {f"S{k}": synthetic_frame(n=2500, drift=0.0003 * (k - 1), seed=k) for k in range(3)}
    p = {**default_params(), "entry_threshold": 0.8}
    ts = frames["S0"].index
    start, end = int(ts[1500]), int(ts[-1]) + 1
    one = portfolio.run(frames, p, start_ts=start, end_ts=end, with_curve=True)
    state, trades, t0 = dict(FRESH), [], start
    while t0 < end:
        t1 = min(end, t0 + 7 * 3_600_000)            # odd chunk: positions and limits cross the cut
        r = portfolio.run(frames, p, start_ts=t0, end_ts=t1, with_curve=True, resume=state)
        state, t0 = r["state"], t1
        trades += r["trade_log"]
    assert len(one["trade_log"]) > 5
    assert [(t["symbol"], t["ts"], round(t["pnl"], 6)) for t in one["trade_log"]] == \
           [(t["symbol"], t["ts"], round(t["pnl"], 6)) for t in trades]


def test_brain2_has_no_order_code():
    src = (Path(__file__).resolve().parent.parent / "core" / "brain2.py").read_text(encoding="utf-8")
    for word in ("broker", "place_order", "create_order", "Account", "notify"):
        assert word not in src
