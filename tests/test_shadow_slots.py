"""With a two-week shadow test, a new candidate must never push out a shadow
that is still being tested."""
import os
import tempfile

from core import shadow
from core.storage import Storage


def test_new_shadow_waits_for_a_free_place():
    st = Storage(os.path.join(tempfile.mkdtemp(), "t.db"))
    ids = [shadow.add(st, {"x": i}, {})["id"] for i in range(shadow.MAX_SHADOWS)]
    assert shadow.add(st, {"x": 99}, {}) is None
    assert [s["id"] for s in shadow.get(st)] == ids          # the running tests are all kept


def test_shadow_clearly_behind_live_after_two_weeks_frees_its_place(monkeypatch):
    st = Storage(os.path.join(tempfile.mkdtemp(), "t.db"))
    now = shadow.now_ms()
    st.set("shadows", [
        {"id": "behind", "created": now - 15 * shadow.DAY_MS, "params": {}, "backtest": {}},
        {"id": "young", "created": now - 10 * shadow.DAY_MS, "params": {}, "backtest": {}},
        {"id": "close", "created": now - 15 * shadow.DAY_MS, "params": {"close": 1}, "backtest": {}}])
    live = {"score": 0.9, "trades": 40, "return": 0.015}
    forward = {"behind": {"score": -2.4, "trades": 23, "return": -0.006},
               "young": {"score": -2.4, "trades": 23, "return": -0.006},
               "close": {"score": 0.8, "trades": 30, "return": 0.012}}
    monkeypatch.setattr(shadow, "load_params", lambda: {"live": 1})
    monkeypatch.setattr(shadow, "clamp_params", lambda p: p)
    ids = iter(["behind", "behind", "young", "young", "close", "close"])

    def fake_forward(frames, p, since):
        sid = next(ids)
        return live if p == {"live": 1} else forward[sid]

    monkeypatch.setattr(shadow.validation, "forward", fake_forward)
    shadow.review(st, {})
    assert [s["id"] for s in shadow.get(st)] == ["young", "close"]   # too young / not clearly behind: kept


def test_shadow_test_is_two_weeks():
    assert shadow.SHADOW_MIN_DAYS == 14 and shadow.MAX_AGE_DAYS > shadow.SHADOW_MIN_DAYS
