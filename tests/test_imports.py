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
