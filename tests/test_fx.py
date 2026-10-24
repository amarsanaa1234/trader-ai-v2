"""FX plug pieces that need no network: trading hours, dollar orientation, swap
sign and the per-market risk bounds."""
from datetime import datetime, timezone

import pandas as pd
import pytest

import markets
from core import context, risk_guard as rg
from core.storage import Storage
from markets.fx import data
from markets.fx.adapter import MARKET as FX


def ms(y, m, d, h):
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)


def test_trading_hours():
    assert FX.is_open(ms(2026, 10, 14, 12))        # Wednesday
    assert not FX.is_open(ms(2026, 10, 16, 21))    # Friday after the close
    assert not FX.is_open(ms(2026, 10, 17, 12))    # Saturday
    assert not FX.is_open(ms(2026, 10, 18, 20))    # Sunday before the open
    assert FX.is_open(ms(2026, 10, 18, 21))        # Sunday open


def test_usd_side():
    assert data.usd_side("USD_JPY") == 1 and data.usd_side("EUR_USD") == -1 and data.usd_side("EUR_GBP") == 0


@pytest.fixture
def fx_db(tmp_path):
    prev = context.mkt()
    context.activate(FX)
    yield Storage(tmp_path / "fx.db")
    context.activate(prev)


def test_swap_charged_once_a_day_and_triple_on_wednesday(fx_db):
    # long pays 2.44%/yr, short receives 0.42%/yr -> symmetric long cost (0.0042+0.0244)/2
    fx_db.db.execute("INSERT INTO financing VALUES ('EUR_USD', -0.0244, 0.0042, '1,1,3,1,1,0,0', 0)")
    idx = pd.Index(range(ms(2026, 10, 12, 0), ms(2026, 10, 17, 0), 3_600_000))   # Mon..Fri, hourly
    h = pd.Series(data._hold_cost(fx_db, "EUR_USD", idx, "1h"), index=idx)
    charged = h[h != 0]
    assert len(charged) == 5                                   # one rollover per weekday
    assert (charged > 0).all()                                 # longs pay on this pair
    daily = (0.0042 + 0.0244) / 2 / 365
    assert sorted(charged.round(12).unique()) == [round(daily, 12), round(3 * daily, 12)]


def test_no_entry_rollover_and_friday():
    idx = pd.Index([ms(2026, 10, 14, 16), ms(2026, 10, 14, 20), ms(2026, 10, 14, 12), ms(2026, 10, 16, 16)])
    # 4h bars closing 20:00 (ok), 00:00 (ok), 16:00 (ok); the 1h config is not used here
    blocked = FX.no_entry(idx)
    assert blocked.tolist() == [False, False, False, True]      # Friday bar closing 20:00 is blocked


def test_weekend_close_and_spread_costs(tmp_path, monkeypatch):
    idx = pd.Index([ms(2026, 10, 16, 12), ms(2026, 10, 16, 16), ms(2026, 10, 14, 16)])
    assert FX.force_exit(idx, {"weekend_close": 0.0}).tolist() == [False, False, False]
    assert FX.force_exit(idx, {"weekend_close": 1.0}).tolist() == [False, True, False]   # Fri bar closing 20:00
    monkeypatch.setattr(FX, "data_dir", tmp_path)
    (tmp_path / "spreads.json").write_text('{"GBP_JPY": 0.0002}', encoding="utf-8")
    assert FX.costs_for("GBP_JPY")["maker"] == pytest.approx(0.0001)
    assert {k: v for k, v in FX.costs_for("EUR_USD").items() if k != "fill_through"} == FX.costs  # not measured -> default
    assert FX.costs_for("GBP_JPY")["fill_through"] == pytest.approx(0.0002 * 0.5)   # half the spread


def test_risk_ladder_steps():
    from core import risk_ladder as rl
    cfg = {"levels": [0.005, 0.01], "window": 30, "up_avg_r": 0.0, "up_profit_factor": 1.2, "down_avg_r": 0.0}
    st, msg = rl.step({}, [0.5] * 10, cfg)
    assert st["level"] == 0 and msg is None                     # not 30 trades yet
    good = [1.0, -0.5] * 15                                     # avg +0.25R, profit factor 2
    st, msg = rl.step(st, good, cfg)
    assert st["level"] == 1 and "1.00%" in msg
    st, msg = rl.step(st, good + [0.1] * 5, cfg)
    assert st["level"] == 1 and msg is None                     # only 5 new trades: wait
    bad = [-1.0, 0.5] * 15 + good                               # newest 30 average -0.25R
    st, msg = rl.step(st, bad, cfg)
    assert st["level"] == 0 and rl.current(st, cfg) == 0.005
    assert max(cfg["levels"]) <= rg.RISK_PER_TRADE


def test_worst_situations_needs_clear_evidence():
    from core import mistakes
    stats = {"against_cot": [-1.0, -0.5, -1.0, 0.2] * 10,      # 40 trades, clearly losing
             "day_mon": [-1.0, 1.0] * 20,                       # coin flip
             "few": [-1.0] * 5}                                 # too few to judge
    found = mistakes.worst_situations(stats)
    assert [f["tag"] for f in found] == ["against_cot"]
    assert mistakes.worst_situations(stats, already=["against_cot"]) == []


def test_blackout_only_for_the_pairs_currencies(fx_db, monkeypatch):
    from core import events
    now = ms(2026, 10, 14, 12)
    fx_db.set("calendar", [{"ts": now, "title": "RBA rate decision", "country": "AUD"}])
    monkeypatch.setattr(events, "MANUAL", events.MANUAL.with_name("no_such_events.json"))
    assert events.blackout(fx_db, now, "AUD_USD")
    assert events.blackout(fx_db, now, "EUR_USD") is None


def test_macro_values_only_after_publication(fx_db):
    from markets.fx import macro
    day = 86_400_000
    t0 = ms(2026, 1, 5, 0)
    fx_db.db.executemany("INSERT INTO rates VALUES (?, ?, ?)",
                         [("AUD", t0, 4.0), ("USD", t0, 3.0), ("AUD", t0 + 10 * day, 4.5)])
    fx_db.db.executemany("INSERT INTO cot VALUES (?, ?, ?)", [("AUD", t0, 0.2), ("JPY", t0, -0.1)])
    at = pd.Index([t0 - day, t0 + day, t0 + 11 * day])
    assert macro.rate_diff(fx_db, "AUD_USD", at).round(6).tolist()[1:] == [1.0, 1.5]
    assert pd.isna(macro.rate_diff(fx_db, "AUD_USD", at).iloc[0])          # nothing public yet
    assert macro.cot_net(fx_db, "AUD_JPY", at).round(6).tolist()[1] == 0.3   # 0.2 - (-0.1)
    assert macro.cot_net(fx_db, "USD_JPY", at).round(6).tolist()[1] == 0.1   # USD counts as 0


def test_month_end_fires_in_the_last_two_business_days(fx_db):
    """Regression: a time-unit slip once made this source silently always 0."""
    step = 4 * 3_600_000
    spx_ts = list(range(ms(2026, 8, 25, 0), ms(2026, 10, 1, 0), step))  # includes the month's start price
    rows = [(t, 100 + k * 0.1, 100 + k * 0.1, 100 + k * 0.1, 100 + k * 0.1, 1) for k, t in enumerate(spx_ts)]
    fx_db.insert_candles("SPX500_USD", "4h", rows)                      # stocks rose all month
    ts = list(range(ms(2026, 9, 1, 0), ms(2026, 10, 1, 0), step))
    v = data._month_end(fx_db, "EUR_USD", "4h", pd.Index(ts))
    closes = pd.to_datetime(pd.Index(ts) + step, unit="ms", utc=True)
    fired = closes[v != 0]
    assert len(fired) > 0 and set(fired.day) <= {29, 30}               # Sep 29-30 2026 (Tue, Wed)
    assert (v[v != 0] == 1.0).all()            # funds sell USD after a stock rally -> EUR_USD up


def test_fx_risk_bounds(fx_db):
    lev, lo, hi = rg.market_bounds()
    assert lev == 3 and lo == pytest.approx(0.0008)
    # a 0.15% stop is refused for crypto but fine for FX; size stays within 3x leverage
    d = rg.check_entry(equity=10_000, entry=1.1, stop=1.1 * (1 - 0.0015), side=1, open_positions=0,
                       open_notional=0, trades_today=0, halted=False, peak_equity=10_000)
    assert d.allowed and d.qty * 1.1 <= 10_000 * 3 + 1e-6
    context.activate(markets.get("crypto"))
    assert not rg.check_entry(equity=10_000, entry=1.1, stop=1.1 * (1 - 0.0015), side=1, open_positions=0,
                              open_notional=0, trades_today=0, halted=False, peak_equity=10_000).allowed
