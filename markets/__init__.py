"""Market registry: markets.get("crypto") -> the crypto Market adapter."""
import importlib

AVAILABLE = ("crypto", "fx", "commodities")


def get(name: str):
    if name not in AVAILABLE:
        raise SystemExit(f"Unknown market {name!r}. Available: {', '.join(AVAILABLE)}")
    return importlib.import_module(f"markets.{name}.adapter").MARKET
