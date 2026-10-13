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


def test_fx_risk_bounds(fx_db):
    lev, lo, hi = rg.market_bounds()
    assert lev == 10 and lo == pytest.approx(0.0008)
    # a 0.15% stop is refused for crypto but fine for FX
    d = rg.check_entry(equity=10_000, entry=1.1, stop=1.1 * (1 - 0.0015), side=1, open_positions=0,
                       open_notional=0, trades_today=0, halted=False, peak_equity=10_000)
    assert d.allowed and d.qty * 1.1 <= 10_000 * 10 + 1e-6
    context.activate(markets.get("crypto"))
    assert not rg.check_entry(equity=10_000, entry=1.1, stop=1.1 * (1 - 0.0015), side=1, open_positions=0,
                              open_notional=0, trades_today=0, halted=False, peak_equity=10_000).allowed
