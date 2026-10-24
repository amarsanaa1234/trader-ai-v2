"""With a month-long shadow test, a new candidate must never push out a shadow
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


def test_shadow_test_is_a_month():
    assert shadow.SHADOW_MIN_DAYS == 30 and shadow.MAX_AGE_DAYS > shadow.SHADOW_MIN_DAYS
