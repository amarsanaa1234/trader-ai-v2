"""Which market this process works on.

Each service runs one market (crypto, fx, ...). main.py calls activate()
once at start-up; core modules then ask mkt() for symbols, paths, costs and
the market's own signal sources, instead of hard-coding any of them."""
_active = None


def activate(market) -> None:
    global _active
    _active = market


def mkt():
    if _active is None:
        raise RuntimeError("No market is active: call core.context.activate(markets.get(name)) first")
    return _active
