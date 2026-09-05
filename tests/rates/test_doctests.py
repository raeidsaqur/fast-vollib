"""Every example in the rates package is executed.

Same runner as the instruments, simulation, and pricing packages.  Non-recursive
by design; ``test_the_package_is_flat_so_nothing_is_skipped`` fails if that ever
stops being safe.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil
from types import ModuleType

import pytest

import fast_vollib.rates as rates


def _package_modules() -> list[ModuleType]:
    modules = [rates]
    for info in pkgutil.iter_modules(rates.__path__):
        modules.append(importlib.import_module(f"{rates.__name__}.{info.name}"))
    return modules


def test_the_package_is_flat_so_nothing_is_skipped() -> None:
    subpackages = [info.name for info in pkgutil.iter_modules(rates.__path__) if info.ispkg]
    assert subpackages == [], (
        f"{subpackages} would not have their doctests run; switch this module to "
        f"pkgutil.walk_packages, as tests/test_surface/test_doctests.py does."
    )


def test_the_examples_are_not_vacuous() -> None:
    """A runner that found no examples would pass while checking nothing."""
    total = sum(len(doctest.DocTestFinder().find(module)) for module in _package_modules())
    assert total > 0


@pytest.mark.parametrize("module", _package_modules(), ids=lambda m: m.__name__)
def test_module_doctests(module: ModuleType) -> None:
    results = doctest.testmod(module, verbose=False, raise_on_error=True)
    del results  # raise_on_error turns any failure into an exception
