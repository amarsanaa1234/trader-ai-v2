"""Commodities futures curve: contract expiry rules, carry from future vs spot,
the paper account's swap and the swap filter."""
import numpy as np
import pandas as pd
import pytest

from markets.commodities import curve
from markets.fx.broker import swap_days


def T(s):
    return pd.Timestamp(s)


def test_front_expiry_rules():
    d = T("2026-10-20")
    assert curve.front_expiry("BCO_USD", d) == T("2026-10-30")     # Brent: last business day of the month
    assert curve.front_expiry("NATGAS_USD", d) == T("2026-10-28")  # gas: 3rd-last business day
    assert curve.front_expiry("BCO_USD", T("2026-10-31")) == T("2026-11-30")
    assert curve.front_expiry("XAU_USD", d) == T("2026-12-29")     # gold: Dec contract
    assert curve.front_expiry("XPT_USD", d) == T("2027-01-27")     # platinum: Jan (Oct already rolled)
    assert curve.front_expiry("CORN_USD", d) == T("2026-12-14")    # corn: business day before the 15th
    assert curve.front_expiry("CORN_USD", T("2026-11-30")) == T("2027-03-12")


def test_carry_sign_and_size():
    days = pd.date_range("2026-10-01", "2026-10-20", freq="B")
    spot = pd.Series(4.60, index=days)
    fut = pd.Series(471.25, index=days)                 # corn in cents, above spot = contango
    c = curve.carry_series("CORN_USD", fut, spot).dropna()
    assert -0.25 < c.iloc[-1] < -0.10                   # about -16% a year, like OANDA's swap
    back = curve.carry_series("BCO_USD", pd.Series(108.0, index=days), pd.Series(109.4, index=days))
    assert back.dropna().iloc[-1] > 0                   # future below spot = backwardation


def test_swap_days():
    mon = int(pd.Timestamp("2026-10-19 12:00", tz="America/New_York").timestamp() * 1000)
    tue = int(pd.Timestamp("2026-10-20 12:00", tz="America/New_York").timestamp() * 1000)
    thu = int(pd.Timestamp("2026-10-22 12:00", tz="America/New_York").timestamp() * 1000)
    week = "1,1,3,1,1,0,0"
    assert swap_days(mon, mon + 3_600_000, week) == 0   # no rollover in between
    assert swap_days(mon, tue, week) == 1
    assert swap_days(mon, thu, week) == 1 + 1 + 3       # Wednesday covers the weekend


def test_swap_block_and_curve_source():
    from markets.commodities.adapter import MARKET as m
    df = pd.DataFrame({"carry": [0.5, -0.5, 0.0, np.nan]})
    assert m.side_filter(df, {"swap_block": 1.0}) is None
    long_bad, short_bad = m.side_filter(df, {"swap_block": 0.3})
    assert list(long_bad) == [False, True, False, False]   # long pays 50% a year in contango
    assert list(short_bad) == [True, False, False, False]  # short pays 50% in backwardation
    assert "curve" in m.source_names and m.default_params["w_curve"] == 0.0
    assert m.default_params["swap_block"] == 0.1
