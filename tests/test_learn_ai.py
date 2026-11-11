"""AI helpers of the 15m learner (core/learn_ai.py): the safety check, news risk tags, and
AI ideas that must pass the evidence test before the learner uses them."""
import numpy as np

from core import ai, learn_ai, orb_learn
from core.storage import Storage
from tests.test_orb15 import START
from tests.test_orb_learn import trade

H = 3_600_000


def test_safety_blocks_events_and_extreme_risk():
    t = trade(10, "USD_JPY:london", 0.5)
    ctx = learn_ai.Context(events=[{"ts": t["entry_ts"] + 30 * 60_000, "title": "Non-Farm Payrolls", "country": "USD"}],
                           risk=[{"ts": t["entry_ts"] - H, "levels": {"USD": "normal", "JPY": "extreme"}}])
    assert ctx.event_near(t["entry_ts"], "USD_JPY") == "USD Non-Farm Payrolls"
    assert ctx.event_near(t["entry_ts"], "EUR_GBP") is None
    assert ctx.risk_at(t["entry_ts"], "USD_JPY") == "extreme"
    assert ctx.risk_at(t["entry_ts"] + 9 * H, "USD_JPY") is None          # too old to count
    book = orb_learn.replay([t], START, ctx=ctx)["book"]
    assert book[0]["weight"] == 0 and "risk=extreme" in book[0]["blocked_by"]
    assert any(b.startswith("event: USD Non-Farm") for b in book[0]["blocked_by"])


def test_ai_idea_is_kept_only_if_it_loses_in_both_halves(tmp_path, monkeypatch):
    st = Storage(tmp_path / "fx.db")
    rng = np.random.default_rng(4)
    known = []
    for k in range(900):
        ny, hot = k % 3 == 0, k % 2 == 0
        r = rng.normal(-0.9 if (ny and hot) else 0.2, 0.8)
        known.append({**trade(k, f"P{k % 5}:{'ny' if ny else 'london'}", float(r)), "vol": 1.5 if hot else 1.0,
                      "side": -1 if k % 4 == 1 else 1})    # shorts never fall in the losing NY+hot group
    ideas = {"ideas": [{"a": "session=ny", "b": "vol=high", "why": "US open whipsaws in volatile markets"},
                       {"a": "dow=2", "b": "side=short", "why": "a guess that should fail"}]}
    monkeypatch.setattr(ai, "enabled", lambda: True)
    monkeypatch.setattr(ai, "ask_json", lambda system, user, schema: ideas)
    monkeypatch.setattr(ai, "model_name", lambda: "test")
    now = known[-1]["exit_ts"] + H
    new = learn_ai.research(st, known, now)
    by = {"+".join(h["tags"]): h for h in new}
    assert by["session=ny+vol=high"]["status"] == "accepted"
    assert by["dow=2+side=short"]["status"] == "rejected"
    assert learn_ai.research(st, known, now + H) is None                   # once a week
    # the learner applies the accepted idea only from its acceptance on
    later = [{**trade(1000 + k, "P9:ny", -0.5, day=900 + k), "vol": 1.5} for k in range(10)]
    ctx = learn_ai.Context.load(st)
    book = orb_learn.replay(later, START, known, ctx)["book"]
    assert all("session=ny+vol=high" in t["blocked_by"] for t in book[1:])


def test_news_risk_is_rated_and_logged(tmp_path, monkeypatch):
    st = Storage(tmp_path / "fx.db")
    monkeypatch.setattr(ai, "enabled", lambda: True)
    monkeypatch.setattr(ai, "model_name", lambda: "test")
    monkeypatch.setattr(ai, "ask_json", lambda s, u, schema: {"currencies": [
        {"ccy": "USD", "level": "high", "reason": "CPI tomorrow"}, {"ccy": "XXX", "level": "low", "reason": "x"}]})
    rec = learn_ai.assess_risk(st, START)
    assert rec["levels"] == {"USD": "high"} and st.get("learn_risk")[-1]["ts"] == START
    assert learn_ai.assess_risk(st, START + H) is None                      # every 4 hours
