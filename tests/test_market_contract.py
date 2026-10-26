"""Every market plug must honour the same contract. When a new market is added
to markets.AVAILABLE these checks run for it automatically."""
import pytest

import markets
from markets.base import Market, tf_to_ms


@pytest.mark.parametrize("name", markets.AVAILABLE)
def test_market_contract(name):
    m = markets.get(name)
    assert isinstance(m, Market) and m.name == name
    assert m.symbols, "no symbols configured"
    assert m.mode in ("signal", "demo", "testnet", "live")
    tf_to_ms(m.timeframe)
    assert set(m.costs) >= {"maker", "taker", "slippage"}
    for s in m.source_names:            # every market source has a default weight
        assert f"w_{s}" in m.default_params, s
    for k in m.param_space:             # every tunable number has a default
        assert k in m.default_params, k
    assert m.data_dir.name == name      # each market keeps its data apart
    for _, (fn, text) in m.commands().items():
        assert callable(fn) and text


def test_timeframes():
    assert tf_to_ms("1h") == 3_600_000
    assert tf_to_ms("15m") == 900_000
    assert tf_to_ms("1d") == 86_400_000
    with pytest.raises(ValueError):
        tf_to_ms("1y")


def test_every_market_module_imports():
    """A missing import inside a market package once stopped the crypto bot loading candles
    (2026-10-26: Storage in markets/crypto/extra.py), so import every module of every market."""
    import importlib
    import pkgutil
    for name in markets.AVAILABLE:
        pkg = importlib.import_module(f"markets.{name}")
        for mod in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"markets.{name}.{mod.name}")
