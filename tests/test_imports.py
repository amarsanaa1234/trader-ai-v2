"""Every module must at least import: a syntax slip in a file no test touched once
crash-looped all the bots (2026-10-19)."""
import importlib
import pkgutil

import pytest

import core
import markets

MODULES = [f"core.{m.name}" for m in pkgutil.iter_modules(core.__path__)] + \
          [f"markets.{n}.adapter" for n in markets.AVAILABLE]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_close_is_recorded_only_when_the_position_is_gone(monkeypatch):
    from types import SimpleNamespace
    import core.trader as tr
    cls = next(c for c in vars(tr).values() if isinstance(c, type) and hasattr(c, "close_trade"))
    monkeypatch.setattr(tr, "notify", lambda *a, **k: None)
    finished = []

    class Broker:
        def __init__(self, stays):
            self.stays = stays

        def close(self, sym):
            pass

        def positions(self):
            return {"SOL": {}} if self.stays else {}

    for stays, expect in ((True, False), (False, True)):
        me = SimpleNamespace(broker=Broker(stays), acc=SimpleNamespace(name="x"), to=lambda: None,
                             finish_trade=lambda t, r: finished.append(r))
        assert cls.close_trade(me, {"symbol": "SOL"}, "flip") is expect
    assert finished == ["flip"]
