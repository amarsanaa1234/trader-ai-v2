"""The hard limits must hold whatever the strategy or the improver does."""
import pytest

from core import risk_guard as rg


def entry(**kw):
    args = dict(equity=1000.0, entry=100.0, stop=98.0, side=1, open_positions=0, open_notional=0.0,
                trades_today=0, halted=False, peak_equity=1000.0)
    args.update(kw)
    return rg.check_entry(**args)


def test_risk_per_trade_is_the_market_cap():
    d = entry()
    assert d.allowed
    cap = rg.market_risk_cap()                 # crypto: 0.5% (owner, 2026-10-19), never above 1%
    assert cap == pytest.approx(0.005) and cap <= rg.RISK_PER_TRADE
    assert d.qty * 2.0 == pytest.approx(1000 * cap)  # stop 2 away -> lose 0.5% of 1000


def test_stop_must_be_on_the_losing_side():
    assert not entry(stop=101.0).allowed
    assert not entry(side=-1, stop=99.0).allowed


def test_stop_distance_bounds():
    assert not entry(stop=99.9).allowed   # 0.1% = noise
    assert not entry(stop=90.0).allowed   # 10% = too wild


def test_leverage_cap():
    d = entry(stop=99.5)  # 1% risk would need 2000 notional; cap is 2x equity
    assert d.qty * 100.0 <= 2 * 1000.0 + 1e-6


def test_halts_and_pauses():
    assert not entry(halted=True).allowed
    assert not entry(paused=True).allowed
    assert not entry(blackout="CPI").allowed


def test_drawdown_scaling_and_halt():
    assert rg.drawdown_multiplier(900, 1000) == 0.5
    assert rg.drawdown_multiplier(790, 1000) == 0.25
    assert rg.drawdown_multiplier(690, 1000) == 0.0
    assert not entry(equity=690.0, peak_equity=1000.0).allowed


def test_correlated_trade_gets_half_risk():
    assert entry(max_corr_same_side=0.9).qty == pytest.approx(entry().qty / 2)


def test_live_limits_are_stricter():
    live, paper = rg.limits("live"), rg.limits("demo")
    assert live["max_open_positions"] == 2 and paper["max_open_positions"] == 10
    assert live["max_total_open_risk"] < paper["max_total_open_risk"]


def test_daily_loss():
    assert rg.daily_loss_hit(1000, 949)
    assert not rg.daily_loss_hit(1000, 960)


def test_live_needs_paper_period_and_confirmation():
    assert not rg.live_trading_allowed(3, rg.LIVE_CONFIRM_PHRASE)[0]
    assert not rg.live_trading_allowed(20, "")[0]
    assert rg.live_trading_allowed(20, rg.LIVE_CONFIRM_PHRASE)[0]
